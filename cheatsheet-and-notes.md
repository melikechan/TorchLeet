# Cheatsheet and Notes

Mostly sourced from docs, with some blog posts and other references.

## Tensor Operations etc

### [torch.squeeze](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html)

`torch.squeeze(input, dim: int | List[int] | None)`

Returns another tensor where all _(or a given list of)_ `1`-sized dimensions are removed in the `input` tensor.

Doesn't affect dimensions that are not `1`-sized.

**Returned tensor shares storage with the `input` tensor.**

```python
>>> x = torch.zeros(1, 1, 3, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])

>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([3, 2])

>>> y = torch.squeeze(x, 2)
>>> y.size()
torch.Size([1, 1, 3, 1, 2])

>>> y = torch.squeeze(x, (0, 1))
>>> y.size()
torch.Size([3, 1, 2])
```

### [torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html)

`torch.unsqueeze(input, dim)`

Adds a size `1` dimension inserted at the specified position (negative dimensions can be used).

**Returned tensor shares storage with the `input` tensor.**

```python
>>> x = torch.zeros(3, 1, 2)
>>> x.size()
torch.Size([3, 1, 2])

>>> y = torch.unsqueeze(x, 1)
>>> y.size()
torch.Size([3, 1, 1, 2])

>>> z = torch.unsqueeze(x, -1)
>>> z.size()
torch.Size([3, 1, 2, 1])
```

[Squeeze vs Unsqueeze](https://stackoverflow.com/questions/61598771/squeeze-vs-unsqueeze-in-pytorch)

### [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)

`Tensor.view(*shape)`

Returns a new tensor with the **same data** but of a different shape.

```python
>>> d = torch.arange(5.)
>>> d
tensor([0., 1., 2., 3., 4.])

>>> e = d.view(-1, 1)
>>> e
tensor([[0.],
        [1.],
        [2.],
        [3.],
        [4.]])

>>> e[2][0] = 6
>>> e
tensor([[0.],
        [1.],
        [6.],
        [3.],
        [4.]])

>>> d
tensor([0., 1., 6., 3., 4.])
```

### [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)

`torch.reshape(input, shape)`

Returns a tensor with the same data and number of elements as input, but with the specified shape.

- Returns a `view` **if possible.**
- Otherview, returns a `copy`.

**You should not depend on the copying vs. viewing behavior.**

```python
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])

>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
```

- [`reshape` vs `view`](https://stackoverflow.com/stack-content/questions/49643225/what-s-the-difference-between-reshape-and-view-in-pytorch)
- [`reshape` vs `view` (another blog post)](https://discuss.pytorch.org/t/whats-the-difference-between-torch-reshape-vs-torch-view/159172)
- [A warning about `reshape` and `view`](https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524)

### [torch.where](https://docs.pytorch.org/docs/stable/generated/torch.where.html)

`torch.where(condition, input, other)`

Returns a tensor of elements selected from either `input` (if `condition` is satisfied) or `other` (if `condition` is not satisfied), depending on condition.

```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])

>>> threshold = 3

>>> torch.where(x <= threshold, 0, x)
tensor([[0, 0, 0],
        [4, 5, 6]])

>>> torch.where(x > threshold, x - 1, x)
tensor([[1, 2, 3],
        [3, 4, 5]])  
```

## Model Stuff

- [Saving / Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) is very explanatory about how to save checkpoints.

## Losses

- [Cross Entropy Loss and Softmax](https://discuss.pytorch.org/t/pytorch-cross-entropy-loss-and-softmax/223498)
- [Don't apply softmax to output while using CrossEntropyLoss](https://stackoverflow.com/a/55675428)
