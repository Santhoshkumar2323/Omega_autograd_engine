from autograd.tensor import Tensor


def checkpoint(fn, *args: Tensor) -> Tensor:
    detached_args = [Tensor(a.data.copy(), requires_grad=False) for a in args]
    forward_only_out = fn(*detached_args)
    out_data = forward_only_out.data.copy()

    req = any(a.requires_grad for a in args)
    out = Tensor(out_data, requires_grad=req, _children=tuple(args), _op="checkpoint")

    def _backward():
        recompute_args = [Tensor(a.data.copy(), requires_grad=a.requires_grad) for a in args]
        recomputed_out = fn(*recompute_args)
        recomputed_out.backward(out.grad)
        for orig, recomputed in zip(args, recompute_args):
            if orig.requires_grad:
                orig.grad += recomputed.grad

    out._backward = _backward
    return out