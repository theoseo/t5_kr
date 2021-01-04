import gin
import t5
import abc
import os
import hashlib
from typing import Iterable, List, Optional, Sequence

import tensorflow.compat.v2 as tf
import tensorflow_text as tf_text


import sentencepiece as sentencepiece_processor
#from t5.data import vocabularies
from t5.data.dataset_providers import Feature
from t5.data import preprocessors
from t5.data import TaskRegistry
import functools

DEFAULT_KR_VOCAB_PATH = "/home/jovyan/work/t5data/tokenizers/t5-notag-vocab.txt"  # GCS
DEFAULT_KR_DATA_PATH = "gs://t5large/data/t5_multi_mecab.txt"
DEFAULT_SPM_PATH = "gs://t5large/data/mecab_sp/tok_32k.model"
DEFAULT_EXTRA_IDS = 100
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
class Vocabulary(metaclass=abc.ABCMeta):
  """Abstract class for all vocabularies.
  Subclasses must implement methods for converting between strings and tokens
  both in pure python (`_encode`/`_decode`) and in TensorFlow
  (`_encode_tf`/`_decode_tf`).
  Subclasses are responsible for reserving PAD_ID=0 as well as EOS_ID and UNK_ID
  if `use_eos` and `use_unk` are True, respectively.
  `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
  """

  def __init__(
      self,
      extra_ids: int = 0,
      use_eos: bool = True,
      use_unk: bool = True):
    """Vocabulary constructor.
    Args:
      extra_ids: The number of extra IDs to reserve.
      use_eos: Whether to stop decoding at EOS_ID=1.
      use_unk: Whether to replace tokens out of range with UNK_ID=2.
    """
    self._extra_ids = extra_ids
    self._use_eos = use_eos
    self._use_unk = use_unk

  @property
  def eos_id(self) -> Optional[int]:
    return EOS_ID if self._use_eos else None

  @property
  def pad_id(self) -> int:
    return PAD_ID

  @property
  def unk_id(self) -> Optional[int]:
    return UNK_ID if self._use_unk else None

  @property
  def extra_ids(self) -> int:
    return self._extra_ids

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including extra ids."""
    return self._base_vocab_size + self.extra_ids

  @abc.abstractproperty
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
    raise NotImplementedError

  @abc.abstractmethod
  def _encode(self, s: str) -> Sequence[int]:
    raise NotImplementedError

  def encode(self, s: str) -> Sequence[int]:
    """Tokenizes string to an int sequence, without adding EOS."""
    return self._encode(s)

  @abc.abstractmethod
  def _decode(self, ids):
    raise NotImplementedError

  def decode(self, ids: Iterable[int]):
    """Detokenizes int32 iterable to a string, up through first EOS."""
    clean_ids = list(ids)

    if self.unk_id is not None:
      clean_ids = [
          self.unk_id if i >= self._base_vocab_size else i
          for i in clean_ids
      ]

    if self.eos_id is not None and self.eos_id in clean_ids:
      clean_ids = clean_ids[:clean_ids.index(self.eos_id) + 1]

    return self._decode(clean_ids)

  @abc.abstractmethod
  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
    return self._encode_tf(s)

  @abc.abstractmethod
  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 N-D Tensor to string (N-1)-D Tensor, through first EOS.
    """
    clean_ids = ids

    if self.unk_id is not None:
      clean_ids = tf.where(
          tf.less(clean_ids, self._base_vocab_size), clean_ids, self.unk_id)

    if self.eos_id is not None:
      # Replace everything after the first EOS_ID with PAD_ID.
      after_eos = tf.cumsum(
          tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
          exclusive=True, axis=-1)
      clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

    return self._decode_tf(clean_ids)

class KorVocabulary(Vocabulary):
    
    def __init__(self, vocab_file, extra_ids=None):
        
        self._vocab_file = vocab_file
        self._tokenizer = None
        

        kwargs = {"extra_ids": extra_ids} if extra_ids is not None else {}
        super().__init__(**kwargs)
        
    @property
    def tokenizer(self):
        if not self._tokenizer:
            mecab_t5_notag =  KoNLPyT5Tokenizer(
                konlpy_wordpiece = KoNLPyWordPieceTokenizer(Mecab(), use_tag=False),
                vocab_file = self._vocab_file
            )

            self._tokenizer = mecab_t5_notag
        return self._tokenizer
        
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    def encode(self, s):
        return self.tokenizer.encode(s)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def _encode_tf(self, s):
        if tf.is_tensor(s):
            s = s.numpy().decode('utf-8')
        encoded_value = self.tokenizer.encode(s)     
        return [encoded_value]
    
    def encode_tf(self, s):
        return tf.py_function(func=self._encode_tf, inp=[s], Tout=tf.int32)

    def decode_tf(self, ids):
        return tf.py_function(func=self.decode, inp=[ids], Tout=tf.string)

class SentencePieceVocabulary(Vocabulary):
  """Wrapper for nlp/sentencepiece encoder.
  Assumes the model was built using flags to reserve ID=0 for padding, ID=1 for
  EOS, and ID=2 for UNK.
  """

  def __init__(self, sentencepiece_model_file, extra_ids=None):
    """Create a SentencePieceVocabulary.
    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels.
    Args:
      sentencepiece_model_file: a string
      extra_ids: an optional integer
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._tokenizer = None
    self._sp_model = None
    super().__init__(use_eos=True, use_unk=True, extra_ids=extra_ids)

  def _load_model(self):
    """Load SPM and Python tokenizer."""
    # Handle cases where SP can't load the file, but gfile can.
    with tf.io.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
      self._sp_model = f.read()
    # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
    # TODO(adarob): Add support for arbitrary EOS and PAD IDs.
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    if self._tokenizer.pad_id() != 0:
      raise ValueError(
          f"Vocabulary PAD ID must be 0, got {self._tokenizer.pad_id()}")
    if self._tokenizer.eos_id() != 1:
      raise ValueError(
          f"Vocabulary EOS ID must be 1, got {self._tokenizer.eos_id()}")
    if self._tokenizer.unk_id() != 2:
      raise ValueError(
          f"Vocabulary UNK ID must be 2, got {self._tokenizer.unk_id()}")

  @property
  def sp_model(self):
    """Retrieve the SPM."""
    if self._sp_model is None:
      self._load_model()
    return self._sp_model

  @property
  def sentencepiece_model_file(self):
    return self._sentencepiece_model_file

  @property
  def tokenizer(self):
    """Returns the Python tokenizer."""
    if not self._tokenizer:
      self._load_model()
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    """Instantiate and return a TF tokenizer."""
    return tf_text.SentencepieceTokenizer(model=self.sp_model)


  @property
  def _base_vocab_size(self):
    """Number of ids (including 0=PAD, 1=EOS, and 2=UNK).
    Returns:
      an integer, the vocabulary size
    """
    return self.tokenizer.GetPieceSize()

  def _encode(self, s):
    """Encode a python string as a list of integers.
    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    return self.tokenizer.EncodeAsIds(s)

  def _decode(self, ids):
    """Decode a list of integers to a python string.
    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    # convert all the extra ids (sentinels) to UNK=2
    ids = [
        self.tokenizer.unk_id() if i >= self.tokenizer.GetPieceSize()
        else i for i in ids]
    return self.tokenizer.DecodeIds(ids)

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.
    This will be necessary for on-the-fly tokenization.
    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    return self.tf_tokenizer.tokenize(s)

  def _decode_tf(self, ids):
    """Decode in TensorFlow.
    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    return self.tf_tokenizer.detokenize(ids)

  def __eq__(self, other):
    try:
      their_md5 = hashlib.md5(other.sp_model).hexdigest()
      their_extra_ids = other.extra_ids
    # If other has no sp_model/extra_ids attribute, we can't test for equality
    except AttributeError:
      return False
    our_md5 = hashlib.md5(self.sp_model).hexdigest()
    return our_md5 == their_md5 and self.extra_ids == their_extra_ids    
    
@gin.configurable
def get_kr_vocabulary(sp_model_path, extra_ids=DEFAULT_EXTRA_IDS):
    return SentencePieceVocabulary(sp_model_path, extra_ids)

    #return KorVocabulary(DEFAULT_KR_VOCAB_PATH, DEFAULT_EXTRA_IDS)

def kr_dataset_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset([DEFAULT_KR_DATA_PATH])
    ds = ds.map(lambda x: {'text': x })
    return ds


default_vocabulary = get_kr_vocabulary(DEFAULT_SPM_PATH)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=default_vocabulary, add_eos=True, required=False),
    "targets": Feature(vocabulary=default_vocabulary, add_eos=True)
}


TaskRegistry.remove('kr_iid_denoising')
TaskRegistry.add(
    "kr_iid_denoising",
    dataset_fn=kr_dataset_fn,
    splits=['train'],
    text_preprocessor=functools.partial(
        preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=preprocessors.iid_denoising,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


FINETUNE_OUTPUT_FEATURES = Feature(vocabulary=default_vocabulary)
BASE_DIR = "gs://t5kornlu"

nli_tsv_path = {
    "train": os.path.join("gs://t5kornlu/nli/data", "mecab_nli_train.tsv"),
    "validation": os.path.join("gs://t5kornlu/nli/data", "mecab_nli_dev.tsv"),
}

def get_nli_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(nli_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds

def nli_preprocessor(ds):
    def normalize_text(text):
        """Lowercase"""
        text = tf.strings.lower(text)
        return text
        

    def to_inputs_and_targets(ex):
        return {
            "inputs":
                tf.strings.join(
                    ["nli ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["targets"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

TaskRegistry.remove('nli')
TaskRegistry.add(
    "nli",
    dataset_fn=get_nli_fn,
    splits=['train', 'validation'],
    text_preprocessor=[nli_preprocessor],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.spearman_corrcoef])                      

korquad_tsv_path = {
    "train": os.path.join("gs://t5kornlu/korquad1.0/data", "korquad_mecab_train.tsv"),
    "validation": os.path.join("gs://t5kornlu/korquad1.0/data", "korquad_mecab_dev.tsv"),
}

def get_korquad_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(korquad_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds

def korquad_preprocessor(ds):
    def normalize_text(text):
        """Lowercase"""
        text = tf.strings.lower(text)
        return text
        

    def to_inputs_and_targets(ex):
        return {
            "inputs":
                tf.strings.join(
                    ["korquad ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["targets"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

TaskRegistry.remove('korquad1.0')
TaskRegistry.add(
    "korquad1.0",
    dataset_fn=get_korquad_fn,
    splits=['train', 'validation'],
    text_preprocessor=[korquad_preprocessor],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy])

stsb_tsv_path = {
    "train": os.path.join("gs://t5kornlu/stsb/data", "mecab_stsb_train.tsv"),
    "validation": os.path.join("gs://t5kornlu/stsb/data", "mecab_stsb_dev.tsv"),
}

def get_stsb_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(stsb_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds

def stsb_preprocessor(ds):
    def normalize_text(text):
        """Lowercase"""
        text = tf.strings.lower(text)
        return text
        

    def to_inputs_and_targets(ex):
        return {
            "inputs":
                tf.strings.join(
                    ["stsb ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["targets"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)  

TaskRegistry.remove('stsb')
TaskRegistry.add(
    "stsb",
    dataset_fn=get_stsb_fn,
    splits=['train', 'validation'],
    text_preprocessor=[stsb_preprocessor],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.spearman_corrcoef])

hate_tsv_path = {
    "train": os.path.join("gs://t5kornlu/hate-speech/data", "mecab_hate_train.tsv"),
    "validation": os.path.join("gs://t5kornlu/hate-speech/data", "mecab_hate_dev.tsv"),
}

def get_hate_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(hate_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds

def hate_preprocessor(ds):
    def normalize_text(text):
        """Lowercase"""
        text = tf.strings.lower(text)
        return text
        

    def to_inputs_and_targets(ex):
        return {
            "inputs":
                tf.strings.join(
                    ["hatespeech: ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["targets"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

TaskRegistry.remove('hate')
TaskRegistry.add(
    "hate",
    dataset_fn=get_hate_fn,
    splits=['train', 'validation'],
    text_preprocessor=[hate_preprocessor],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy, \
                t5.evaluation.metrics.sklearn_metrics_wrapper("f1_score", average="macro")])

nsmc_tsv_path = {
    "train": os.path.join("gs://t5kornlu/nsmc/data", "mecabsp_ratings_train.tsv"),
    "validation": os.path.join("gs://t5kornlu/nsmc/data", "mecabsp_ratings_test.tsv"),
}

def get_nsmc_fn(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(nsmc_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds

def nsmc_preprocessor(ds):
    def normalize_text(text):
        """Lowercase"""
        text = tf.strings.lower(text)
        return text
        

    def to_inputs_and_targets(ex):
        return {
            "inputs":
                tf.strings.join(
                    ["nsmc: ", normalize_text(ex["inputs"])]),
            "targets": normalize_text(ex["targets"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
TaskRegistry.remove('nsmc')
TaskRegistry.add(
    "nsmc",
    dataset_fn=get_nsmc_fn,
    splits=['train', 'validation'],
    text_preprocessor=[nsmc_preprocessor],
    output_features=FINETUNE_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy])
