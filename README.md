# 60th Place Solution for the Stanford Ribonanza RNA Folding Competition

## Context

* Competition overview: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/overview

* Data: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data

## Summary 

The notebook published by Iafoss [1] was my starting point. 
I used transformer architecture in this competition. 
Here are the details of the generated submissions that 
might have contributed to the result:

* In addition to sequence itself, I used minimum free energy (MFE) 
RNA secondary structure 
in dot-parentheses notation. 
For some experiments, I also calculated base pair probability matrices 
(BPPs), 
however, the best-scoring submission did not use them. 
Both MFE and BPPs were obtained via EternaFold package [2].
* Position information is encoded via rotary embeddings [3] in the best-scoring 
submission. Some other experiments use both rotary and 
sinusoidal embeddings.
* Sequences and MFE indices always have end-of-sequence (EOS) token 
appended at the beginning and at the end.
* Sequences and MFE indices are padded from both sides so that 
the middles of all RNA molecules enter the model at roughly 
the same position.

## Scores

The best model achieved 0.15867 MAE on private 
and 0.15565 MAE on public leaderboard. I will refer to this submission as 
"submission-27" for historical reasons.

"Submission-23" was the second submission I chose for evaluation, with 
0.17211 private and 0.15363 public MAE score.

The submission with the best public score was not chosen for evaluation 
because the model does not generalize well, as per "How to check if 
your model generalizes to long sequences" discussion.

## Models

### "Submission-27"

The model for "submission-27" consists of one decoder layer as the 
first layer
and eight BERT-like encoder layers on top. 

The decoder layer accepts 
sequence embeddings 
as "target" and MFE-structure embeddings as "memory"
(information that would be coming from an encoder in a 
traditional encoder-decoder transformer). A mask that prevents the 
decoder from attending to padding tokens is applied. Other than masking 
the padding tokens, no other mask is applied, and the decoder is able 
to attend to all information 
it receives.

Rotary embeddings are used in every attention module inside the model.

A schematic presentation of the architecture is given in Figure 1.

### "Submission-23"



## Training

The training procedure was adopted from the starter notebook. 
The training was performed for 45 epochs on 
"Quick_Start" data. 







## Notes on the code

This repository is set up to reproduce two of my solutions: 
23 and 27 (numbered this way for historical reasons).

## Other approaches that were tried



## Acknowledgements


## References










