# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

## This files gives basic tensor library functionality, because yes we can
import strformat, macros, sequtils, random

type
  Tensor[Rank: static[int], T] = object
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the tensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: A data storage for the tensor
    ##   - Rank is part of the type for optimization purposes
    ##
    ## Warning ⚠:
    ##   Assignment ```var a = b``` does not copy the data. Data modification on one tensor will be reflected on the other.
    ##   However modification on metadata (shape, strides or offset) will not affect the other tensor.
    shape: array[Rank, int]
    strides: array[Rank, int]
    offset: int
    storage: CpuStorage[T]

  CpuStorage*{.shallow.}[T] = object
    ## Data storage for the tensor, copies are shallow by default
    data*: seq[T]

template tensor(result: var Tensor, shape: array) =
  result.shape = shape

  var accum = 1
  for i in countdown(Rank - 1, 0):
    result.strides[i] = accum
    accum *= shape[i]

func newTensor*[Rank: static[int], T](shape: array[Rank, int]): Tensor[Rank, T] =
  tensor(result, shape)
  result.storage.data = newSeq[T](shape.product)

proc randomTensor*[Rank: static[int], T](shape: array[Rank, int], max: T): Tensor[Rank, T] =
  tensor(result, shape)
  result.storage.data = newSeqWith(shape.product, T(rand(max)))

func getIndex[Rank, T](t: Tensor[Rank, T], idx: array[Rank, int]): int {.inline.} =
  ## Convert [i, j, k, l ...] to the memory location referred by the index
  result = t.offset
  for i in 0 ..< t.Rank:
    {.unroll.} # I'm sad this doesn't work yet
    result += t.strides[i] * idx[i]

func `[]`[Rank, T](t: Tensor[Rank, T], idx: array[Rank, int]): T {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)]

func `[]=`[Rank, T](t: var Tensor[Rank, T], idx: array[Rank, int], val: T) {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)] = val

template `[]`[T: SomeNumber](x: T, idx: varargs[int]): T =
  ## "Index" scalars
  x

func shape(x: SomeNumber): array[1, int] = [1]

func bcShape[R1, R2: static[int]](x: array[R1, int]; y: array[R2, int]): auto =
  when R1 > R2:
    result = x
    for i, idx in result.mpairs:
      if idx == 1 and y[i] != 1:
        idx = y[i]
  else:
    result = y
    for i, idx in result.mpairs:
      if idx == 1 and x[i] != 1:
        idx = x[i]

macro getBroadcastShape(x: varargs[typed]): untyped =
  assert x.len >= 2
  result = nnkDotExpr.newTree(x[0], ident"shape")
  for i in 1 ..< x.len:
    let xi = x[i]
    result = quote do: bcShape(`result`, `xi`.shape)

func bc[R1, R2: static[int], T](t: Tensor[R1, T], shape: array[R2, int]): Tensor[R2, T] =
  ## Broadcast tensors
  result.shape = shape
  for i in 0 ..< R1:
    if t.shape[i] == 1 and shape[i] != 1:
      result.strides[i] = 0
    else:
      result.strides[i] = t.strides[i]
      if t.shape[i] != result.shape[i]:
        raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")
  result.offset = t.offset
  result.storage = t.storage

func bc[Rank; T: SomeNumber](x: T, shape: array[Rank, int]): T {.inline.}=
  ## "Broadcast" scalars
  x

func product(x: varargs[int]): int =
  result = 1
  for val in x: result *= val

proc replaceNodes(ast: NimNode, values: NimNode, containers: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - an array of replacement value
  #   - an array of identifiers to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      for i, c in containers:
        if node.eqIdent($c):
          return values[i]
      return node
    of nnkEmpty:
      return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

proc pop*(tree: var NimNode): NimNode =
  ## varargs[untyped] consumes all arguments so the actual value should be popped
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

func nb_elems[N: static[int], T](x: typedesc[array[N, T]]): static[int] =
  N

macro broadcast(inputs_body: varargs[untyped]): untyped =
  var
    inputs = inputs_body
    body = inputs.pop()

  let
    shape = genSym(nskLet, "broadcast_shape__")
    coord = genSym(nskVar, "broadcast_coord__")

  var doBroadcast = newStmtList()
  var bcInputs = nnkArgList.newTree()
  for input in inputs:
    let broadcasted = genSym(nskLet, "broadcast_" & $input & "__")
    doBroadcast.add newLetStmt(
      broadcasted,
      newCall(ident"bc", input, shape)
    )
    bcInputs.add nnkBracketExpr.newTree(broadcasted, coord)

  body = body.replaceNodes(bcInputs, inputs)

  result = quote do:
    block:
      let `shape` = getBroadcastShape(`inputs`)
      const rank = `shape`.type.nb_elems
      var `coord`: array[rank, int] # Current coordinates in the n-dimensional space
      `doBroadcast`

      var output = newTensor[rank, type(`body`)](`shape`)
      var counter = 0

      while counter < `shape`.product:
        # Assign for the current iteration
        output[`coord`] = `body`

        # Compute the next position
        for k in countdown(rank - 1, 0):
          if `coord`[k] < `shape`[k] - 1:
            `coord`[k] += 1
            break
          else:
            `coord`[k] = 0
        inc counter

      # Now return the value
      output

when isMainModule:
  let x = randomTensor([1, 2, 3], 10)
  let y = randomTensor([5, 2], 10)

  echo x
  echo y

  let a = broadcast(x, y):
    x * y

  echo a
