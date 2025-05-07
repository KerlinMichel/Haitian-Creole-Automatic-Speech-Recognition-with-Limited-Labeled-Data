# Haitian Creole Automatic Speech Recognition with Limited Labeled Data

Haitian Creole is a low-resource language in the context of creating a model for Automatic Speech Recognition (ASR). The largest dataset that exists is [Carnegie Mellon University's School of Computer Science's (CMU SCS) Haitian Creole Speech dataset](http://www.speech.cs.cmu.edu/haitian/) which has < 20 hours of transcribed Haitian Creole speech whereas a high-resource language could have several hundreds of hours of transcribed speech.

Wav2Vec2 overcomes the challenges of training a model to perform ASR for a low-resource language by using unsupervised pre-training on a vast amount of unlabeled speech data.
To improve the output of the trained Wav2Vec2 ASR model a Language Model (LM) is used to rescore the ASR model outputs. The LM used is Kenneth Heafield's [KenLM](https://github.com/kpu/kenlm) and [transcription text](http://www.speech.cs.cmu.edu/haitian/) and [text only data](http://www.speech.cs.cmu.edu/haitian/text/) from CMU SCS is used to create a Haitian Creole LM.

The Haitian Creole ASR model was used to create the first free automatic Speech Translation between Haitian Creole speech and English speech available at:

* [Play Store (Android)](https://play.google.com/store/apps/details?id=com.traduikreyol.traduiapp&hl=en_US)

## Choosing the right ASR model
<details>
<summary>Click here to expand</summary>

Three popular models were experimented with: OpenAI's Whisper, Facebook AI's XLSR-Wav2Vec2, using the Wav2Vec 2.0 model, and w2v-BERT 2.0.

The model output quality of Whisper was significantly better than XLSR-Wav2Vec2 for high-resource languages but the reverse was true for Haitian Creole, a low-resource language. Quality was measured using Word Error Rate (WER).

The [XLSR-Wav2Vec2 model is pretrained on 56,000 hours of speech data spanning 53 languages which includes Haitian Creole](https://arxiv.org/pdf/2006.13979). XLSR-Wav2Vec2 needs to be fine-tuned since XLSR-Wav2Vec2 is just the model that represents human speech.

XLSR-Wav2Vec2 with and without a LM outperformed a Whisper model fine-tuned for Haitian Creole. In general Whisper outperforms Wav2Vec 2.0 models so it is likely that it is possible that increasing the amount of transcribed Haitian Creole speech data to some point would make the Whisper model have a lower WER than the XLSR-Wav2Vec2 model. But since transcribing hundreds of hours of speech data is very laborious and Whisper has a much higher inference time Wav2Vec2 models were chosen to create a Haitian Creole ASR model. w2v-BERT 2.0 is a direct improvement from the Wav2Vec 2.0 model and was pre-trained on millions of hours of speech data.

w2v-BERT 2.0 was used to create a Haitian Creole Translation model.

See the following for further discussion on the differences between Wav2Vec 2.0 and Whisper: https://deepgram.com/learn/benchmarking-top-open-source-speech-models
</details>

## Data Preparation

### Normalizing Transcription Text
To make model training easier transcription text should be normalized. "one", "1" and "One." are different transcriptions that could be correct for some speech audio. Normalizing those transcriptions "one" would make training the model easier. In Haitian Creole "tap" is a contraction of "t ap" and is sometimes use interchangeably in transcriptions and sound similar with the only difference being a brief paused between words in "t ap". "tap" is mapped to "t ap" to normalize that word.

To effectively normalize some corpus of text an understanding of the language is needed which is why I created a [Haitian Creole NLP library](https://github.com/KerlinMichel/kreyol_nlp) to process Haitian Creole text.

Text Normalization
* lowercase all text
* expand all contractions
* map transcription using digits to number words (e.g. "9" and "nèf" maps to "nèf")
* remove punctuations and other special characters

## Training ASR model

Model was trained by following https://huggingface.co/blog/fine-tune-w2v2-bert.

The alphabets used to train the Wav2Vec2ForCTC are:

##### Multigraph Alphabet
```
⟨a⟩, ⟨an⟩, ⟨b⟩, ⟨ch⟩, ⟨d⟩,  ⟨e⟩,  ⟨è⟩,   ⟨en⟩,
⟨f⟩, ⟨g⟩,  ⟨h⟩, ⟨i⟩,  ⟨j⟩,  ⟨k⟩,  ⟨l⟩,   ⟨m⟩,
⟨n⟩, ⟨ng⟩, ⟨o⟩, ⟨ò⟩,  ⟨on⟩, ⟨ou⟩, ⟨oun⟩, ⟨p⟩
⟨r⟩, ⟨s⟩,  ⟨t⟩, ⟨ui⟩, ⟨v⟩,  ⟨w⟩,  ⟨y⟩,   ⟨z⟩
```
##### Unigraph Alphabet
```
⟨a⟩, ⟨b⟩, ⟨c⟩, ⟨d⟩, ⟨e⟩, ⟨è⟩, ⟨f⟩, ⟨g⟩,
⟨h⟩, ⟨i⟩, ⟨j⟩, ⟨k⟩, ⟨l⟩, ⟨m⟩, ⟨n⟩, ⟨o⟩,
⟨ò⟩, ⟨p⟩, ⟨r⟩, ⟨s⟩, ⟨t⟩, ⟨u⟩, ⟨v⟩, ⟨w⟩,
⟨y⟩, ⟨z⟩
```
A lower WER was seen using the multigraph alphabet so that was used for the final model.

### Validation Data
As a native speaker I created a small (< 10 minutes of speech) [Haitian Creole Transcription dataset](https://github.com/KerlinMichel/KreyolTranskripsyon/tree/main/data) and use it as a validation dataset. This validation dataset is useable but would be a better measure of how well a model is trained if more data were added and have more than one speaker.

### Language Model
A LM using https://github.com/kpu/kenlm was created to rescore outputs of the w2v-BERT 2.0 model. The corpus used to created to LM is available through the following code: https://github.com/KerlinMichel/kreyol_nlp/blob/main/kreyol_nlp/corpus/\_\_init\_\_.py

TorchAudio's [CTC Decoder](https://docs.pytorch.org/audio/main/generated/torchaudio.models.decoder.CTCDecoder.html), used to map the ASR models output to transcriptions, includes a feature to use a language model for rescoring. 

### Results

#### Inference on validation data
| ASR Model                                      | WER     |
| --------                                       | ------- |
| w2v-BERT 2.0 (Greedy CTC)                      | 0.42    |
| w2v-BERT 2.0 (text normalization and LM)       | 0.27    |

#### Example ASR model outputs

| w2v-BERT 2.0 (text normalization and LM) output | Ground Truth |  Ground Truth English Translation
| --------                                        | -------      | ------- |
| ou ap kont jòj | ou ap konn jòj | You'll know George (Haitian saying)
| ayiti se yon zile | ayiti se yon zile | Haiti is an island
| aprè fi manje li rive etidye jiska ke li wotè   | apre ou fin manje li liv ou e etidye jiskaske li uit è | After you are done eating read your book until it is eight'o clock

## Model Inference
The Haitian Creole ASR model is being used in a translation system that translates between Haitian Creole speech and English speech.

The inference is being done on CPU machines instead of machines with GPUs available to minimize cost of running the system. Since running machine learning models are much slower on CPUs than GPUs the model was optimized using [ONNX](https://onnx.ai/) to run faster on CPUs.

### Convert model to ONNX format code snippet
```python
import torch

model = # pytorch w2v-BERT 2.0 model

dummy_input = torch.randn(1, 250_000, requires_grad=True),

torch.onnx.export(model,
                  dummy_input,
                  'model.onnx',
                  export_params=True,
                  opset_version=14,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {1: 'audio_len'},
                        'output': {1: 'audio_len'}})
```

For more code on how to manage and use a ONNX Wav2Vec style model see https://github.com/ccoreilly/wav2vec2-service/tree/master.

## Live ASR
The trained ASR model can transcribe an entire audio file but to transcribe audio as it's being recorded live then short segments of the live recorded audio needs to be transcribed and then combined together. But splitting the audio into segments reduce the quality of the output since the model lacks context so context should be included by including strides before and after the audio segment being transcribed.

<img title="ASR with Strides" alt="ASR with Strides" src="https://huggingface.co/blog/assets/49_asr_chunking/Striding.png">

The graphic above and a more detail explanation of the striding technique are at https://huggingface.co/blog/asr-chunking.
