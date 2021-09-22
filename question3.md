## Question 3 

#### Infer the sequence of states for sentences 10150-10152 of the Brown corpus and compare against the truth.


The accuracy for the first sentence is 0.77 (10 out of 13).

```python
out:   ['DET', 'VERB', 'ADP', 'ADJ', 'DET', 'VERB', 'NOUN', 'DET', 'VERB', 'PRT', 'VERB', 'VERB', '.']
truth: ['DET', 'VERB', 'ADP', 'ADJ', 'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'PRT', 'VERB', 'VERB', '.']
```

The accuracy for the second sentence is 0.67 (12 out of 18).

```python
out:   ['DET', 'ADP', 'NOUN', 'VERB', 'DET', 'ADP', 'NOUN', 'NOUN', 'ADP', 'DET',
        'ADP', 'NOUN', 'VERB', 'PRON', 'ADP', 'NOUN', 'DET', '.']
truth: ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADJ', 'ADJ', 'NOUN', 'ADP', 'DET',
        'ADJ', 'NOUN', 'VERB', 'VERB', 'ADP', 'NUM', 'DET', '.']
```

The accuracy for the third sentence is 0.81 (13 out of 16).

```python
out:   ['PRON', 'VERB', 'DET', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP',
        'DET', 'ADP', 'NOUN', 'CONJ', '.', 'NOUN', '.']
truth: ['PRON', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP',
        'DET', 'ADJ', 'NOUN', 'CONJ', 'DET', 'NOUN', '.']
```
