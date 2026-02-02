//#region rolldown:runtime
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __commonJS = (cb, mod$1) => function() {
	return mod$1 || (0, cb[__getOwnPropNames(cb)[0]])((mod$1 = { exports: {} }).exports, mod$1), mod$1.exports;
};
var __export = (target, all$1) => {
	for (var name in all$1) __defProp(target, name, {
		get: all$1[name],
		enumerable: true
	});
};
var __copyProps = (to, from, except, desc) => {
	if (from && typeof from === "object" || typeof from === "function") for (var keys = __getOwnPropNames(from), i = 0, n = keys.length, key$1; i < n; i++) {
		key$1 = keys[i];
		if (!__hasOwnProp.call(to, key$1) && key$1 !== except) __defProp(to, key$1, {
			get: ((k) => from[k]).bind(null, key$1),
			enumerable: !(desc = __getOwnPropDesc(from, key$1)) || desc.enumerable
		});
	}
	return to;
};
var __toESM = (mod$1, isNodeMode, target) => (target = mod$1 != null ? __create(__getProtoOf(mod$1)) : {}, __copyProps(isNodeMode || !mod$1 || !mod$1.__esModule ? __defProp(target, "default", {
	value: mod$1,
	enumerable: true
}) : target, mod$1));

//#endregion
const require_backend = require('./backend-BpA-kG3A.cjs');

//#region src/frontend/convolution.ts
/**
* Check that the shapes and parameters passed to convolution are valid.
* Expected shapes of the lhs and rhs of the convolution are:
*
* - `lhsShape = [*vmapDims, batchSize, inChannels, spatialDims...]`
* - `rhsShape = [*vmapDims, outChannels, inChannels, kernelSize...]`
*
* If the check succeeds, returns the output shape.
*/
function checkConvShape(lhsShape, rhsShape, { vmapDims, strides, padding, lhsDilation, rhsDilation }) {
	if (lhsShape.length !== rhsShape.length) throw new Error(`conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`);
	const n = lhsShape.length - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	if (strides.length !== n) throw new Error("conv() strides != spatial dims");
	if (padding.length !== n) throw new Error("conv() padding != spatial dims");
	if (lhsDilation.length !== n) throw new Error("conv() lhsDilation != spatial dimensions");
	if (rhsDilation.length !== n) throw new Error("conv() rhsDilation != spatial dimensions");
	if (lhsShape[vmapDims + 1] !== rhsShape[vmapDims + 1]) throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
	const outShape = [
		...require_backend.generalBroadcast(lhsShape.slice(0, vmapDims), rhsShape.slice(0, vmapDims)),
		lhsShape[vmapDims],
		rhsShape[vmapDims]
	];
	for (let i = 0; i < n; i++) {
		if (strides[i] <= 0 || !Number.isInteger(strides[i])) throw new Error(`conv() strides[${i}] must be a positive integer`);
		if (padding[i].length !== 2 || !padding[i].every(Number.isInteger)) throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
		if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i])) throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
		if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i])) throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);
		const [x, k] = [lhsShape[i + vmapDims + 2], rhsShape[i + vmapDims + 2]];
		if (k <= 0) throw new Error("conv() kernel size must be positive");
		const [pl, pr] = padding[i];
		if (pl < -x || pr < -x || pl + pr < -x) throw new Error(`conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`);
		const kernelSize = (k - 1) * rhsDilation[i] + 1;
		const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
		if (kernelSize > inSize) throw new Error(`conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`);
		outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
	}
	return outShape;
}
function checkPoolShape(inShape, window, strides) {
	if (strides.length !== window.length) throw new Error("pool() strides != window dims");
	if (window.length > inShape.length) throw new Error("pool() window has more dimensions than input");
	const outShape = inShape.slice(0, inShape.length - window.length);
	for (let i = 0; i < window.length; i++) {
		const k = window[i];
		const s = strides[i];
		const size$1 = inShape[inShape.length - window.length + i];
		if (k <= 0 || !Number.isInteger(k)) throw new Error(`pool() window[${i}] must be a positive integer`);
		if (k > size$1) throw new Error(`pool() window[${i}]=${k} > input size ${size$1}`);
		if (s <= 0 || !Number.isInteger(s)) throw new Error(`pool() strides[${i}] must be a positive integer`);
		outShape.push(Math.ceil((size$1 - k + 1) / s));
	}
	return outShape.concat(window);
}
/**
* Takes a shape tracker and a kernel size `ks`, then reshapes it so the last
* `ks.length` dimensions become `2 * ks.length` dimensions by treating them as
* spatial dimensions convolved with a kernel.
*
* The resulting array can be multiplied with a kernel of shape `ks`, then
* reduced along the last `ks.length` dimensions for a convolution.
*
* Reference: https://github.com/tinygrad/tinygrad/blob/v0.10.3/tinygrad/tensor.py#L2097
*/
function pool(st, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (st.shape.length < ks.length) throw new Error("pool() called with too many dimensions");
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	if (strides.some((s) => s <= 0 || !Number.isInteger(s))) throw new Error("pool() strides must be positive integers");
	if (dilation.some((d) => d <= 0 || !Number.isInteger(d))) throw new Error("pool() dilation must be positive integers");
	const noop = st.shape.slice(0, -ks.length);
	const i_ = st.shape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, o * s - i])]);
		st = st.reshape([...noop, ...require_backend.zip(o_, s_).flatMap(([o, s]) => [o, s])]).shrink([...noop.map((x) => [0, x]), ...require_backend.zip(o_, ks).flatMap(([o, k]) => [[0, o], [0, k]])]);
		st = st.permute([
			...require_backend.range(noop.length),
			...ks.map((_, j) => noop.length + 2 * j),
			...ks.map((_, j) => noop.length + 2 * j + 1)
		]);
		return st;
	}
	const f_ = require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
	const kidf = require_backend.zipn(ks, i_, d_, f_);
	st = st.repeat([...require_backend.rep(noop.length, 1), ...kidf.map(([k, i, d, f]) => Math.ceil(k * (i * f + d) / i))]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kidf.map(([k, i, d, f]) => [0, k * (i * f + d)])]).reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o, s]) => [[0, k], [0, o * s]])]).reshape([...noop, ...kos.flat(1)]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o]) => [
		[0, k],
		[0, o],
		[0, 1]
	])]).reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);
	st = st.permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/**
* Perform the transpose of pool, directly undo-ing a pool() operation.
*
* Note that since pool repeats the input, the transpose operation technically
* should include a sum reduction. This function doesn't perform the reduction,
* which should be done on the last `k` axes of the returned shape.
*/
function poolTranspose(st, inShape, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	const noop = inShape.slice(0, -ks.length);
	const i_ = inShape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j])]);
		st = st.pad([...noop.map(() => [0, 0]), ...require_backend.zip(s_, ks).flatMap(([s, k]) => [[0, 0], [0, s - k]])]).reshape([...noop, ...require_backend.zip(o_, s_).map(([o, s]) => o * s)]);
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, i - o * s])]);
		return st.reshape(st.shape.concat(require_backend.rep(ks.length, 1)));
	}
	if (!require_backend.deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length))) throw new Error("poolTranspose() called with mismatched output shape");
	const f_ = require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
	const kidf = require_backend.zipn(ks, i_, d_, f_);
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [
		k,
		o,
		1
	])]).pad([...noop.map(() => [0, 0]), ...s_.flatMap((s) => [
		[0, 0],
		[0, 0],
		[0, s - 1]
	])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([...noop.map(() => [0, 0]), ...kidf.flatMap(([_k, i, d, f], j) => [[0, 0], [0, i * f + d - o_[j] * s_[j]]])]);
	st = st.reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))]).pad([...noop.map(() => [0, 0]), ...kidf.map(([k, i, d, f]) => [0, Math.ceil(k * (i * f + d) / i) * i - k * (i * f + d)])]);
	st = st.reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [Math.ceil(k * (i * f + d) / i), i])]).permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/** Applies dilation to an array directly, for transposed convolution. */
function applyDilation(st, dilation) {
	if (dilation.every((s) => s === 1)) return st;
	const s_ = dilation;
	const n = s_.length;
	const prefix = st.shape.slice(0, -n);
	const k_ = st.shape.slice(-n);
	st = st.reshape([...prefix, ...k_.flatMap((k) => [k, 1])]);
	st = st.pad([...prefix.map(() => [0, 0]), ...s_.flatMap((s) => [[0, 0], [0, s - 1]])]);
	st = st.reshape([...prefix, ...k_.map((k, i) => k * s_[i])]);
	st = st.shrink([...prefix.map((p) => [0, p]), ...k_.map((k, i) => [0, (k - 1) * s_[i] + 1])]);
	return st;
}
/**
* Prepare for a convolution between two arrays.
*
* This does not check the validity of the shapes, which should be checked
* beforehand using `checkConvShape()`.
*/
function prepareConv(stX, stY, params) {
	const v = params.vmapDims;
	const n = stX.shape.length - 2 - v;
	const vmapShape = stX.shape.slice(0, v);
	stX = applyDilation(stX, params.lhsDilation);
	const ks = stY.shape.slice(v + 2);
	stX = stX.padOrShrink([...require_backend.rep(v + 2, [0, 0]), ...params.padding]);
	stX = pool(stX, ks, params.strides, params.rhsDilation);
	stX = stX.moveaxis(v + 1, v + n + 1).reshape([
		...vmapShape,
		stX.shape[v],
		1,
		...stX.shape.slice(v + 2, v + n + 2),
		stX.shape[v + 1] * require_backend.prod(ks)
	]);
	stY = stY.reshape([
		...vmapShape,
		1,
		stY.shape[v],
		...require_backend.rep(n, 1),
		stY.shape[v + 1] * require_backend.prod(ks)
	]);
	return [stX, stY];
}

//#endregion
//#region src/tree.ts
var tree_exports = {};
__export(tree_exports, {
	JsTreeDef: () => JsTreeDef,
	NodeType: () => NodeType,
	dispose: () => dispose,
	flatten: () => flatten,
	leaves: () => leaves,
	map: () => map,
	ref: () => ref,
	structure: () => structure,
	unflatten: () => unflatten
});
const JsArray$2 = globalThis.Array;
let NodeType = /* @__PURE__ */ function(NodeType$1) {
	NodeType$1["Array"] = "Array";
	NodeType$1["Object"] = "Object";
	NodeType$1["Leaf"] = "Leaf";
	return NodeType$1;
}({});
/** Represents the structure of a JsTree. */
var JsTreeDef = class JsTreeDef {
	static leaf = new JsTreeDef(NodeType.Leaf, null, []);
	constructor(nodeType, nodeMetadata, childTreedefs) {
		this.nodeType = nodeType;
		this.nodeMetadata = nodeMetadata;
		this.childTreedefs = childTreedefs;
	}
	/** Get the total number of leaves in the tree. */
	get size() {
		return this.nodeType === NodeType.Leaf ? 1 : this.childTreedefs.reduce((a, b) => a + b.size, 0);
	}
	/** Returns a string representation of this tree definition. */
	toString(root = true) {
		if (root) return "JsTreeDef(" + this.toString(false) + ")";
		switch (this.nodeType) {
			case NodeType.Leaf: return "*";
			case NodeType.Array: return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
			case NodeType.Object: {
				const parts = [];
				for (let i = 0; i < this.childTreedefs.length; i++) parts.push(`${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`);
				return `{${parts.join(", ")}}`;
			}
		}
	}
	/** Compare this tree definition with another. */
	equals(other) {
		return this.nodeType === other.nodeType && require_backend.deepEqual(this.nodeMetadata, other.nodeMetadata) && this.childTreedefs.length === other.childTreedefs.length && this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]));
	}
};
function quoteObjectKey(key$1) {
	if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key$1)) return key$1;
	return JSON.stringify(key$1);
}
/** Flatten a structured object, returning the tree definition. */
function flatten(tree) {
	const leaves$1 = [];
	const treedef = _flatten(tree, leaves$1);
	return [leaves$1, treedef];
}
function _flatten(tree, leaves$1) {
	if (JsArray$2.isArray(tree)) {
		const childTrees = tree.map((c) => _flatten(c, leaves$1));
		return new JsTreeDef(NodeType.Array, null, childTrees);
	} else if (typeof tree === "object" && tree !== null && tree.constructor === Object) {
		const [keys, values] = require_backend.unzip2(Object.entries(tree));
		const childTrees = values.map((c) => _flatten(c, leaves$1));
		return new JsTreeDef(NodeType.Object, keys, childTrees);
	} else {
		leaves$1.push(tree);
		return JsTreeDef.leaf;
	}
}
/** Get the leaves of a tree. */
function leaves(tree) {
	return flatten(tree)[0];
}
/** Get the treedef for a tree. */
function structure(tree) {
	return flatten(tree)[1];
}
/** Reconstruct a structured object from the flattened representation. */
function unflatten(treedef, leaves$1) {
	return _unflatten(treedef, leaves$1[Symbol.iterator]());
}
function _unflatten(treedef, leaves$1) {
	switch (treedef.nodeType) {
		case NodeType.Leaf: {
			const { value, done } = leaves$1.next();
			if (done) throw new TypeError("Ran out of leaves while unflattening JsTree");
			return value;
		}
		case NodeType.Array: return treedef.childTreedefs.map((c) => _unflatten(c, leaves$1));
		case NodeType.Object: {
			const obj = {};
			for (let i = 0; i < treedef.childTreedefs.length; i++) obj[treedef.nodeMetadata[i]] = _unflatten(treedef.childTreedefs[i], leaves$1);
			return obj;
		}
	}
}
/** Maps a multi-input function over pytree args to produce a new pytree. */
function map(fn, tree, ...rest) {
	const [leaves$1, treedef] = flatten(tree);
	const restLeaves = rest.map((x) => flatten(x)[0]);
	const resultLeaves = [];
	for (let i = 0; i < leaves$1.length; i++) resultLeaves.push(fn(leaves$1[i], ...restLeaves.map((x) => x[i])));
	return unflatten(treedef, resultLeaves);
}
/** Take a reference of every array in a tree. */
function ref(tree) {
	return map((x) => x instanceof Tracer ? x.ref : x, tree);
}
/** Dispose every array in a tree. */
function dispose(tree) {
	if (tree) map((x) => x instanceof Tracer ? x.dispose() : void 0, tree);
}

//#endregion
//#region src/frontend/core.ts
/**
* Frontend primitive operations, which are lowered into Kernel objects before
* being dispatched to the backend.
*
* Any operation between arrays can be described in these parts. This is also
* the set of primitives that can occur in Jaxpr programs, and the level at
* which transformations like vmap, grad, and jvp occur. They are loosely based
* on [XLA](https://openxla.org/xla/operation_semantics).
*
* All n-ary operations support broadcasting, with NumPy semantics.
*/
let Primitive = /* @__PURE__ */ function(Primitive$1) {
	Primitive$1["Add"] = "add";
	Primitive$1["Mul"] = "mul";
	Primitive$1["Idiv"] = "idiv";
	Primitive$1["Mod"] = "mod";
	Primitive$1["Min"] = "min";
	Primitive$1["Max"] = "max";
	Primitive$1["Neg"] = "neg";
	Primitive$1["Reciprocal"] = "reciprocal";
	Primitive$1["Floor"] = "floor";
	Primitive$1["Ceil"] = "ceil";
	Primitive$1["StopGradient"] = "stop_gradient";
	Primitive$1["Cast"] = "cast";
	Primitive$1["Bitcast"] = "bitcast";
	Primitive$1["Sin"] = "sin";
	Primitive$1["Cos"] = "cos";
	Primitive$1["Asin"] = "asin";
	Primitive$1["Atan"] = "atan";
	Primitive$1["Exp"] = "exp";
	Primitive$1["Log"] = "log";
	Primitive$1["Erf"] = "erf";
	Primitive$1["Erfc"] = "erfc";
	Primitive$1["Sqrt"] = "sqrt";
	Primitive$1["Reduce"] = "reduce";
	Primitive$1["Dot"] = "dot";
	Primitive$1["Conv"] = "conv";
	Primitive$1["Pool"] = "pool";
	Primitive$1["PoolTranspose"] = "pool_transpose";
	Primitive$1["Compare"] = "compare";
	Primitive$1["Where"] = "where";
	Primitive$1["Concatenate"] = "concatenate";
	Primitive$1["Split"] = "split";
	Primitive$1["RandomBits"] = "random_bits";
	Primitive$1["Gather"] = "gather";
	Primitive$1["Transpose"] = "transpose";
	Primitive$1["Broadcast"] = "broadcast";
	Primitive$1["Reshape"] = "reshape";
	Primitive$1["Flip"] = "flip";
	Primitive$1["Shrink"] = "shrink";
	Primitive$1["Pad"] = "pad";
	Primitive$1["Sort"] = "sort";
	Primitive$1["Argsort"] = "argsort";
	Primitive$1["TriangularSolve"] = "triangular_solve";
	Primitive$1["Cholesky"] = "cholesky";
	Primitive$1["LU"] = "lu";
	Primitive$1["Jit"] = "jit";
	Primitive$1["Scan"] = "scan";
	return Primitive$1;
}({});
let CompareOp = /* @__PURE__ */ function(CompareOp$1) {
	CompareOp$1["Less"] = "less";
	CompareOp$1["Equal"] = "equal";
	CompareOp$1["NotEqual"] = "not_equal";
	CompareOp$1["LessEqual"] = "less_equal";
	return CompareOp$1;
}({});
const routinePrimitives = new Map([
	[Primitive.Sort, require_backend.Routines.Sort],
	[Primitive.Argsort, require_backend.Routines.Argsort],
	[Primitive.TriangularSolve, require_backend.Routines.TriangularSolve],
	[Primitive.Cholesky, require_backend.Routines.Cholesky],
	[Primitive.LU, require_backend.Routines.LU]
]);
function add$1(x, y) {
	return bind1(Primitive.Add, [x, y]);
}
function mul(x, y) {
	return bind1(Primitive.Mul, [x, y]);
}
function idiv(x, y) {
	return bind1(Primitive.Idiv, [x, y]);
}
function mod(x, y) {
	return bind1(Primitive.Mod, [x, y]);
}
function min$1(x, y) {
	return bind1(Primitive.Min, [x, y]);
}
function max$1(x, y) {
	return bind1(Primitive.Max, [x, y]);
}
function neg(x) {
	return bind1(Primitive.Neg, [x]);
}
function reciprocal$1(x) {
	return bind1(Primitive.Reciprocal, [x]);
}
function floor$1(x) {
	return bind1(Primitive.Floor, [x]);
}
function ceil$1(x) {
	return bind1(Primitive.Ceil, [x]);
}
function stopGradient(x) {
	return bind1(Primitive.StopGradient, [x]);
}
function cast(x, dtype) {
	return bind1(Primitive.Cast, [x], { dtype });
}
function bitcast(x, dtype) {
	return bind1(Primitive.Bitcast, [x], { dtype });
}
function sin$1(x) {
	return bind1(Primitive.Sin, [x]);
}
function cos$1(x) {
	return bind1(Primitive.Cos, [x]);
}
function asin$1(x) {
	return bind1(Primitive.Asin, [x]);
}
function atan$1(x) {
	return bind1(Primitive.Atan, [x]);
}
function exp$1(x) {
	return bind1(Primitive.Exp, [x]);
}
function log$1(x) {
	return bind1(Primitive.Log, [x]);
}
function erf$1(x) {
	return bind1(Primitive.Erf, [x]);
}
function erfc$1(x) {
	return bind1(Primitive.Erfc, [x]);
}
function sqrt$1(x) {
	return bind1(Primitive.Sqrt, [x]);
}
function reduce(x, op, axis = null, opts) {
	if (!require_backend.AluGroup.Reduce.has(op)) throw new TypeError(`Invalid reduce operation: ${op}`);
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	const originalShape = getShape(x);
	let result = bind1(Primitive.Reduce, [x], {
		op,
		axis
	});
	if (opts?.keepdims) result = result.reshape(originalShape.map((dim, i) => axis.includes(i) ? 1 : dim));
	return result;
}
function dot$2(x, y) {
	return bind1(Primitive.Dot, [x, y]);
}
function conv$1(x, y, params = {}) {
	if (x.ndim !== y.ndim) throw new Error(`conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`);
	const vmapDims = params.vmapDims ?? 0;
	const n = x.ndim - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	return bind1(Primitive.Conv, [x, y], {
		vmapDims,
		strides: params.strides ?? require_backend.rep(n, 1),
		padding: params.padding ?? require_backend.rep(n, [0, 0]),
		lhsDilation: params.lhsDilation ?? require_backend.rep(n, 1),
		rhsDilation: params.rhsDilation ?? require_backend.rep(n, 1)
	});
}
function compare(x, y, op) {
	return bind1(Primitive.Compare, [x, y], { op });
}
function greater$1(x, y) {
	return compare(y, x, CompareOp.Less);
}
function less$1(x, y) {
	return compare(x, y, CompareOp.Less);
}
function equal$1(x, y) {
	return compare(x, y, CompareOp.Equal);
}
function notEqual$1(x, y) {
	return compare(x, y, CompareOp.NotEqual);
}
function greaterEqual$1(x, y) {
	return compare(y, x, CompareOp.LessEqual);
}
function lessEqual$1(x, y) {
	return compare(x, y, CompareOp.LessEqual);
}
function where$1(cond, x, y) {
	return bind1(Primitive.Where, [
		cond,
		x,
		y
	]);
}
function concatenate$1(xs, axis) {
	if (xs.length === 0) throw new Error("concatenate requires at least one input");
	const avals = xs.map((x) => ShapedArray.fromAval(getAval(x)));
	axis = require_backend.checkAxis(axis, avals[0].ndim);
	for (const x of avals) if (x.ndim !== avals[0].ndim || !x.shape.every((s, i) => i === axis || s === avals[0].shape[i])) throw new Error(`Concatenate: inputs ${avals[0]} and ${x} must match shapes except on axis ${axis}`);
	return bind1(Primitive.Concatenate, xs, { axis });
}
function split$2(x, axis, sizes) {
	axis = require_backend.checkAxis(axis, ndim$1(x));
	if (sizes.some((s) => s < 0 || !Number.isInteger(s))) throw new Error(`split: sizes must be nonnegative integers, got ${JSON.stringify(sizes)}`);
	const totalSize = sizes.reduce((a, b) => a + b, 0);
	if (totalSize !== getShape(x)[axis]) throw new Error(`split: sizes must sum to the size of the axis ${axis}, got ${totalSize}`);
	return bind(Primitive.Split, [x], {
		axis,
		sizes
	});
}
function randomBits(k0, k1, shape$1, mode = "xor") {
	if (!require_backend.deepEqual(k0.shape, k1.shape) || k0.dtype !== require_backend.DType.Uint32 || k1.dtype !== require_backend.DType.Uint32) throw new Error(`randomBits: key parts must be uint32 with the same shape, got ${ShapedArray.fromAval(k0.aval)} and ${ShapedArray.fromAval(k1.aval)}`);
	return bind1(Primitive.RandomBits, [k0, k1], {
		shape: shape$1,
		mode
	});
}
function gather(x, indices, axis, outDim) {
	if (indices.length === 0) throw new Error("gather() requires at least one index");
	if (!Array.isArray(axis) || axis.length !== indices.length) throw new Error(`Invalid gather() axis: expected ${indices.length} axes, got ${JSON.stringify(axis)}`);
	axis = axis.map((a) => require_backend.checkAxis(a, ndim$1(x)));
	if (new Set(axis).size !== axis.length) throw new Error(`Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`);
	outDim = require_backend.checkAxis(outDim, ndim$1(x) - axis.length + 1);
	return bind1(Primitive.Gather, [x, ...indices], {
		axis,
		outDim
	});
}
function transpose$1(x, perm) {
	perm = perm ? perm.map((a) => require_backend.checkAxis(a, ndim$1(x))) : require_backend.range(ndim$1(x)).reverse();
	if (!require_backend.isPermutation(perm, ndim$1(x))) throw new Error(`Invalid transpose permutation for ${ndim$1(x)} axes: ${JSON.stringify(perm)}`);
	return bind1(Primitive.Transpose, [x], { perm });
}
function broadcast(x, shape$1, axis) {
	axis = require_backend.normalizeAxis(axis, shape$1.length);
	return bind1(Primitive.Broadcast, [x], {
		shape: shape$1,
		axis
	});
}
function reshape$1(x, shape$1) {
	if (typeof shape$1 === "number") shape$1 = [shape$1];
	const originalShape = getShape(x);
	const autoIdx = shape$1.indexOf(-1);
	if (autoIdx !== -1) {
		const remaining = require_backend.prod(originalShape) / -require_backend.prod(shape$1);
		if (!Number.isInteger(remaining) || remaining < 0) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
		shape$1 = shape$1.toSpliced(autoIdx, 1, remaining);
	}
	if (require_backend.prod(originalShape) !== require_backend.prod(shape$1)) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
	return bind1(Primitive.Reshape, [x], { shape: shape$1 });
}
function flip$1(x, axis) {
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	return bind1(Primitive.Flip, [x], { axis });
}
function shrink(x, slice) {
	const shape$1 = getShape(x);
	if (!Array.isArray(slice) || !slice.every(require_backend.isNumberPair)) throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
	if (slice.length !== shape$1.length) throw new Error(`Invalid shrink(): expected ${shape$1.length} axes, got ${slice.length}`);
	for (let i = 0; i < shape$1.length; i++) {
		const [start, end] = slice[i];
		if (start > end || start < 0 || end > shape$1[i]) throw new Error(`Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape$1[i]}`);
	}
	return bind1(Primitive.Shrink, [x], { slice });
}
function pad$1(x, width) {
	const nd = ndim$1(x);
	let w;
	if (typeof width === "number") w = [[width, width]];
	else if (require_backend.isNumberPair(width)) w = [width];
	else if (!Array.isArray(width)) {
		const indicesAndPairs = Object.entries(width);
		w = require_backend.rep(nd, [0, 0]);
		for (const [k, v] of indicesAndPairs) w[require_backend.checkAxis(parseInt(k), nd)] = v;
	} else if (!width.every(require_backend.isNumberPair)) throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
	else w = width;
	if (w.length === 1) {
		const [w0, w1] = w[0];
		w = require_backend.rep(nd, () => [w0, w1]);
	} else if (w.length !== nd) throw new Error(`Invalid pad(): expected ${nd} axes, got ${w.length}`);
	return bind1(Primitive.Pad, [x], { width: w });
}
function triangularSolve$1(a, b, { lower = false, unitDiagonal = false } = {}) {
	const as = getShape(a);
	const bs = getShape(b);
	if (as.length < 2 || bs.length < 2) throw new Error(`triangular_solve: must be >=2D, got a=${as}, b=${bs}`);
	const n = as[as.length - 2];
	if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
	if (lower) {
		a = flip$1(a, [-2, -1]);
		b = flip$1(b, [-1]);
	}
	let x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
	if (lower) x = flip$1(x, [-1]);
	return x;
}
function cholesky$2(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`cholesky: expected batch of square matrices, got ${aval}`);
	return bind1(Primitive.Cholesky, [x]);
}
function lu$1(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2) throw new Error(`lu: expected batch of matrices, got ${aval}`);
	return bind(Primitive.LU, [x]);
}
function sort$1(x) {
	const nd = ndim$1(x);
	if (nd === 0) throw new Error("sort: requires at least 1D input");
	return bind1(Primitive.Sort, [x]);
}
function argsort$1(x) {
	const nd = ndim$1(x);
	if (nd === 0) throw new Error("argsort: requires at least 1D input");
	return bind(Primitive.Argsort, [x]);
}
function bind1(prim, args, params = {}) {
	const [results] = bind(prim, args, params);
	return results;
}
const traceStack = [];
let dynamicTrace = null;
/**
* Push an interpreter onto the trace stack. Use this like:
* `using main = newMain(...);`
*/
function newMain(traceType, globalData = null) {
	const level = traceStack.length;
	const main = {
		level,
		traceType,
		globalData
	};
	traceStack.push(main);
	return Object.assign(main, { [Symbol.dispose]() {
		traceStack.pop();
	} });
}
/**
* Set the current dynamic trace, which stashes the current interpreter stack
* and acts temporarily as the bottom of the stack. Use this like:
* `using _dynamic = newDynamic(main);`
*/
function newDynamic(main) {
	const prevDynamicTrace = dynamicTrace;
	dynamicTrace = main;
	return { [Symbol.dispose]() {
		dynamicTrace = prevDynamicTrace;
	} };
}
function currentTraceLevel() {
	return traceStack[traceStack.length - 1].level;
}
var Trace = class {
	constructor(main) {
		this.main = main;
	}
};
/**
* Broadcast shapes and promote types with casting for two avals.
*
* This implements the weak type behavior described in `promoteTypes()`, but not
* implemented in that function as `weakType` is not passed.
*/
function promoteAvals(a, b) {
	const shape$1 = require_backend.generalBroadcast(a.shape, b.shape);
	const weakType = a.weakType && b.weakType;
	let dtype;
	if (a.weakType === b.weakType) dtype = require_backend.promoteTypes(a.dtype, b.dtype);
	else if (a.weakType) dtype = require_backend.promoteTypes(b.dtype, require_backend.DType.Uint32);
	else dtype = require_backend.promoteTypes(a.dtype, require_backend.DType.Uint32);
	return new ShapedArray(shape$1, dtype, weakType);
}
var Tracer = class Tracer {
	/** @ignore */
	_trace;
	constructor(trace$1) {
		this._trace = trace$1;
	}
	/** The shape of the array. */
	get shape() {
		return this.aval.shape;
	}
	/** The total number of elements in the array. */
	get size() {
		return require_backend.prod(this.shape);
	}
	/** The dtype of elements stored in the array. */
	get dtype() {
		return this.aval.dtype;
	}
	/**
	* Whether the array is weakly typed.
	*
	* Weakly typed arrays will cast to the dtype of the other operand. See
	* `promoteTypes()` for details.
	*/
	get weakType() {
		return this.aval.weakType;
	}
	/** The number of dimensions of the array. */
	get ndim() {
		return this.shape.length;
	}
	/** @ignore */
	fullLower() {
		return this;
	}
	neg() {
		return neg(this);
	}
	add(other) {
		return add$1(this, other);
	}
	mul(other) {
		return mul(this, other);
	}
	mod(other) {
		return mod(this, other);
	}
	greater(other) {
		return greater$1(this, other);
	}
	less(other) {
		return less$1(this, other);
	}
	equal(other) {
		return equal$1(this, other);
	}
	notEqual(other) {
		return notEqual$1(this, other);
	}
	greaterEqual(other) {
		return greaterEqual$1(this, other);
	}
	lessEqual(other) {
		return lessEqual$1(this, other);
	}
	/** Sum of the elements of the array over a given axis, or axes. */
	sum(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Add, axis, opts);
	}
	/** Product of the array elements over a given axis. */
	prod(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Mul, axis, opts);
	}
	/** Compute the average of the array elements along the specified axis. */
	mean(axis = null, opts) {
		axis = require_backend.normalizeAxis(axis, this.ndim);
		const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
		if (n === 0) throw new Error("mean: cannot compute mean over zero-length axis");
		const originalDtype = this.dtype;
		const castDtype = require_backend.promoteTypes(originalDtype, require_backend.DType.Float32);
		const result = reduce(this.astype(castDtype), require_backend.AluOp.Add, axis, opts);
		return result.mul(1 / n).astype(originalDtype);
	}
	/** Minimum of the elements of the array along a given axis. */
	min(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Min, axis, opts);
	}
	/** Maximum of the elements of the array along a given axis. */
	max(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Max, axis, opts);
	}
	/** Test whether all array elements along a given axis evaluate to true. */
	all(axis = null, opts) {
		return this.astype(require_backend.DType.Bool).min(axis, opts);
	}
	/** Test whether any array element along a given axis evaluates to true. */
	any(axis = null, opts) {
		return this.astype(require_backend.DType.Bool).max(axis, opts);
	}
	/** Permute the dimensions of an array. Defaults to reversing the axis order. */
	transpose(perm) {
		return transpose$1(this, perm);
	}
	/**
	* Give a new shape to an array without changing its data.
	*
	* One shape dimension can be -1. In this case, the value is inferred from the
	* length of the array and remaining dimensions.
	*/
	reshape(shape$1) {
		return reshape$1(this, shape$1);
	}
	/** Copy the array and cast to a specified dtype. */
	astype(dtype) {
		if (this.dtype === dtype) return this;
		return cast(this, dtype);
	}
	/** Subtract an array from this one. */
	sub(other) {
		return this.add(neg(other));
	}
	/** Divide an array by this one. */
	div(other) {
		if (require_backend.isFloatDtype(this.dtype)) return this.mul(reciprocal$1(other));
		return idiv(this, other);
	}
	/** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
	diagonal(offset = 0, axis1 = 0, axis2 = 1) {
		if (!Number.isInteger(offset)) throw new TypeError(`offset must be an integer, got ${offset}`);
		if (offset < 0) return this.diagonal(-offset, axis2, axis1);
		axis1 = require_backend.checkAxis(axis1, this.ndim);
		axis2 = require_backend.checkAxis(axis2, this.ndim);
		if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
		if (offset >= this.shape[axis2]) throw new Error("offset exceeds axis size");
		let ar = this;
		if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
			const perm = require_backend.range(ar.ndim).filter((i) => i !== axis1 && i !== axis2).concat(axis1, axis2);
			ar = ar.transpose(perm);
		}
		const [n, m] = ar.shape.slice(-2);
		const diagSize = Math.min(n, m - offset);
		ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
		const npad = diagSize * (m + 1) - n * m;
		if (npad > 0) ar = pad$1(ar, [...require_backend.rep(ar.ndim - 1, [0, 0]), [0, npad]]);
		else if (npad < 0) ar = shrink(ar, [...ar.shape.slice(0, -1), n * m + npad].map((x) => [0, x]));
		ar = ar.reshape([
			...ar.shape.slice(0, -1),
			diagSize,
			m + 1
		]);
		ar = shrink(ar, [...ar.shape.slice(0, -1).map((x) => [0, x]), [offset, offset + 1]]).reshape(ar.shape.slice(0, -1));
		return ar;
	}
	/** Flatten the array without changing its data. */
	flatten() {
		return this.reshape(-1);
	}
	/** Flatten the array without changing its data. */
	ravel() {
		return this.reshape(-1);
	}
	/**
	* Iterate over the first dimension of this array, returning slices.
	*
	* This can be used to destructure arrays. For example:
	*
	* ```js
	* let x = np.array([[1, 2], [3, 4]]);
	* let [a, b] = x;
	* console.log(a.js()); // [1, 2]
	* console.log(b.js()); // [3, 4]
	* ```
	*/
	*[Symbol.iterator]() {
		if (this.ndim === 0) throw new Error("Cannot iterate over a scalar array");
		let residual = this;
		const subarrayShape = this.shape.slice(1);
		for (let i = 0; i < this.shape[0]; i++) {
			const lr = split$2(residual, 0, [1, residual.shape[0] - 1]);
			yield lr[0].reshape(subarrayShape);
			residual = lr[1];
		}
		residual.dispose();
	}
	/**
	* Return a sorted copy of an array in ascending order.
	*
	* See `jax.numpy.sort` for full docs.
	*/
	sort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (this.shape[axis] <= 1) return this;
		if (axis === this.ndim - 1) return sort$1(this);
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		return sort$1(this.transpose(perm)).transpose(require_backend.invertPermutation(perm));
	}
	/**
	* Return the indices that would sort an array. This may not be a stable
	* sorting algorithm; it need not preserve order of indices in ties.
	*
	* See `jax.numpy.argsort` for full docs.
	*/
	argsort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (axis === this.ndim - 1) return argsort$1(this)[1];
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		return argsort$1(this.transpose(perm))[1].transpose(require_backend.invertPermutation(perm));
	}
	/**
	* Slice an array along one or more axes.
	*
	* This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
	* mimic this in JavaScript, we would write:
	*
	* ```js
	* x.slice([1, 3], 2, [], null);
	* ```
	*
	* The `slice` method accepts a variable number of arguments, each of which
	* can be a number, an empty array, a single-element array, a two-element
	* array, or `null`. The arguments are interpreted as follows:
	*
	* - A number `n` means to access the `n`-th element along that axis, removing
	*   that axis from the resulting shape.
	* - An empty array `[]` means to keep that axis as-is, like `:` in Python.
	* - A single-element array `[i]` means to start slicing from index `i`
	*   (inclusive) to the end of the axis, like `x[i:]`.
	* - A two-element array `[i, j]` means to slice from index `i` (inclusive)
	*   to index `j` (exclusive), like `x[i:j]`.
	* - `null` means to add a new axis at that position, like `np.newaxis`.
	*
	* Like in Python, negative indices are supported, which count from the end of
	* the axis. For example, `-1` means the last element.
	*
	* Strided slices are not yet implemented, so you cannot write `x[::2]` or
	* similar.
	*
	* Advanced indexing by integer arrays is also supported. This translates to
	* the "gather" primitive, and it allows you to access specific elements of
	* the array by integer indices stored in another array.
	*/
	slice(...index) {
		const checkBounds = (n, i) => {
			if (i > n || i < -n) throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
			return i < 0 ? n + i : i;
		};
		const hasAdvancedIdx = index.some((value) => value instanceof Tracer);
		const axesForGather = [];
		let outDim = -1;
		if (hasAdvancedIdx) {
			const advancedAxes = [];
			let currentAxisForGather = 0;
			for (let i = 0; i < index.length; i++) {
				const value = index[i];
				if (value instanceof Tracer) {
					advancedAxes.push(i);
					axesForGather.push(currentAxisForGather++);
				} else if (typeof value === "number") advancedAxes.push(i);
				else currentAxisForGather++;
			}
			if (advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !== advancedAxes.length - 1) outDim = 0;
			else outDim = axesForGather[0];
		}
		const slice = [];
		const basicShape = [];
		let needsReshape = false;
		let axis = 0;
		for (const value of index) if (value === null) {
			basicShape.push(1);
			needsReshape = true;
		} else if (typeof value === "number") {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const i = checkBounds(this.shape[axis++], value);
			slice.push([i, i + 1]);
			needsReshape = true;
		} else if (Array.isArray(value)) {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const n = this.shape[axis++];
			if (value.length === 0) {
				basicShape.push(n);
				slice.push([0, n]);
			} else if (value.length === 1) {
				const i = checkBounds(n, value[0]);
				basicShape.push(n - i);
				slice.push([i, n]);
			} else if (value.length === 2) {
				const [i, j] = value.map((v) => checkBounds(n, v));
				if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
				basicShape.push(j - i);
				slice.push([i, j]);
			}
		} else if (value instanceof Tracer) {
			const n = this.shape[axis++];
			basicShape.push(n);
			slice.push([0, n]);
		} else throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
		while (axis < this.shape.length) {
			slice.push([0, this.shape[axis]]);
			basicShape.push(this.shape[axis++]);
		}
		let result = shrink(this, slice);
		result = needsReshape ? reshape$1(result, basicShape) : result;
		if (hasAdvancedIdx) result = gather(result, index.filter((a) => a instanceof Tracer), axesForGather, outDim);
		return result;
	}
};
function ndim$1(x) {
	if (x instanceof Tracer) return x.shape.length;
	else return 0;
}
function getShape(x) {
	return x instanceof Tracer ? x.shape : [];
}
var ShapedArray = class ShapedArray {
	constructor(shape$1, dtype, weakType) {
		this.shape = shape$1;
		this.dtype = dtype;
		this.weakType = weakType;
	}
	static fromAval(aval) {
		return new ShapedArray(aval.shape, aval.dtype, aval.weakType);
	}
	get ndim() {
		return this.shape.length;
	}
	get size() {
		return require_backend.prod(this.shape);
	}
	scalar() {
		return new ShapedArray([], this.dtype, this.weakType);
	}
	toString() {
		return `${this.dtype}[${this.shape.join(",")}]`;
	}
	equals(other) {
		return this === other || this.constructor === other.constructor && this.ndim === other.ndim && this.shape.every((d, i) => d === other.shape[i]);
	}
};
function getAval(x) {
	if (x instanceof Tracer) return x.aval;
	else if (typeof x === "boolean" || typeof x === "number") return new ShapedArray([], typeof x === "boolean" ? require_backend.DType.Bool : require_backend.DType.Float32, typeof x === "boolean" ? false : true);
	else throw new TypeError(`Unknown value: ${x}`);
}
function bind(prim, args, params = {}) {
	const topTrace = findTopTrace(args);
	const tracers = args.map((arg) => fullRaise(topTrace, arg));
	const outs = topTrace.processPrimitive(prim, tracers, params);
	if (require_backend.DEBUG >= 5) console.info(`processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`);
	return outs.map((out) => out.fullLower());
}
function findTopTrace(xs) {
	let topMain = traceStack[0];
	for (const x of xs) if (x instanceof Tracer && x._trace.main.level > topMain.level) topMain = x._trace.main;
	if (dynamicTrace && dynamicTrace.level > topMain.level) topMain = dynamicTrace;
	return new topMain.traceType(topMain);
}
function fullRaise(trace$1, val) {
	if (!(val instanceof Tracer)) return trace$1.pure(val);
	const level = trace$1.main.level;
	if (Object.is(val._trace.main, trace$1.main)) return val;
	else if (val._trace.main.level < level) return trace$1.lift(val);
	else if (val._trace.main.level > level) throw new Error(`Can't lift Tracer level ${val._trace.main.level} to level ${level}`);
	else throw new Error(`Different traces at same level: ${val._trace.constructor}, ${trace$1.constructor}.`);
}
var TreeMismatchError = class extends TypeError {
	constructor(where$2, left, right) {
		super(`Mismatched tree structures in ${where$2}: ${left} != ${right}`);
	}
};
/** Flatten a function of `JsTree` input/output for use in tracing. */
function flattenFun(f, inTree) {
	const store = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const pytreeArgs = unflatten(inTree, argsFlat);
		const out = f(...pytreeArgs);
		const [outFlat, outTree] = flatten(out);
		store.value = outTree;
		return outFlat;
	};
	return [flatFun, store];
}
/** Like flattenFun, but expects f to return [main, aux] tuple. */
function flattenFunWithAux(f, inTree) {
	const store = { value: void 0 };
	const auxStore = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const pytreeArgs = unflatten(inTree, argsFlat);
		const result = f(...pytreeArgs);
		if (!Array.isArray(result) || result.length !== 2) throw new Error("Function with `hasAux: true` must return [output, aux] tuple");
		const [out, aux] = result;
		const [outFlat, outTree] = flatten(out);
		store.value = outTree;
		auxStore.value = aux;
		return outFlat;
	};
	return [
		flatFun,
		store,
		auxStore
	];
}
var UseAfterFreeError = class extends ReferenceError {
	constructor(tracer) {
		super(`Referenced tracer ${tracer.toString()} freed, please use .ref move semantics`);
	}
};

//#endregion
//#region node_modules/.pnpm/@oxc-project+runtime@0.78.0/node_modules/@oxc-project/runtime/src/helpers/usingCtx.js
var require_usingCtx = /* @__PURE__ */ __commonJS({ "node_modules/.pnpm/@oxc-project+runtime@0.78.0/node_modules/@oxc-project/runtime/src/helpers/usingCtx.js": ((exports, module) => {
	function _usingCtx() {
		var r = "function" == typeof SuppressedError ? SuppressedError : function(r$1, e$2) {
			var n$1 = Error();
			return n$1.name = "SuppressedError", n$1.error = r$1, n$1.suppressed = e$2, n$1;
		}, e$1 = {}, n = [];
		function using(r$1, e$2) {
			if (null != e$2) {
				if (Object(e$2) !== e$2) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
				if (r$1) var o = e$2[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
				if (void 0 === o && (o = e$2[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r$1)) var t = o;
				if ("function" != typeof o) throw new TypeError("Object is not disposable.");
				t && (o = function o$1() {
					try {
						t.call(e$2);
					} catch (r$2) {
						return Promise.reject(r$2);
					}
				}), n.push({
					v: e$2,
					d: o,
					a: r$1
				});
			} else r$1 && n.push({
				d: e$2,
				a: r$1
			});
			return e$2;
		}
		return {
			e: e$1,
			u: using.bind(null, !1),
			a: using.bind(null, !0),
			d: function d() {
				var o, t = this.e, s = 0;
				function next() {
					for (; o = n.pop();) try {
						if (!o.a && 1 === s) return s = 0, n.push(o), Promise.resolve().then(next);
						if (o.d) {
							var r$1 = o.d.call(o.v);
							if (o.a) return s |= 2, Promise.resolve(r$1).then(next, err);
						} else s |= 1;
					} catch (r$2) {
						return err(r$2);
					}
					if (1 === s) return t !== e$1 ? Promise.reject(t) : Promise.resolve();
					if (t !== e$1) throw t;
				}
				function err(n$1) {
					return t = t !== e$1 ? new r(n$1, t) : n$1, next();
				}
				return next();
			}
		};
	}
	module.exports = _usingCtx, module.exports.__esModule = true, module.exports["default"] = module.exports;
}) });

//#endregion
//#region src/frontend/jaxpr.ts
var import_usingCtx$2 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/** Variable in a Jaxpr expression. */
var Var = class Var {
	static #nextId = 1;
	id;
	aval;
	constructor(aval) {
		this.id = Var.#nextId++;
		this.aval = aval;
	}
	toString() {
		return `Var(${this.id}):${this.aval.toString()}`;
	}
};
/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
var Lit = class {
	value;
	aval;
	get dtype() {
		return this.aval.dtype;
	}
	constructor(aval, value) {
		if (aval.shape.length !== 0) throw new Error(`internal: Lit must be a scalar`);
		this.value = value;
		this.aval = ShapedArray.fromAval(aval);
	}
};
function atomIsLit(atom, literal) {
	return atom instanceof Lit && (literal === void 0 || atom.value === literal);
}
var VarPrinter = class {
	names = /* @__PURE__ */ new Map();
	#next = "a";
	#advance() {
		const ret = this.#next;
		let lastNonz = this.#next.length - 1;
		while (lastNonz >= 0 && this.#next[lastNonz] === "z") lastNonz--;
		if (lastNonz < 0) this.#next = "a".repeat(this.#next.length + 1);
		else {
			let result = this.#next.slice(0, lastNonz);
			result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
			result += "a".repeat(this.#next.length - 1 - lastNonz);
			this.#next = result;
		}
		return ret;
	}
	name(v) {
		if (this.names.has(v)) return this.names.get(v);
		const name = this.#advance();
		this.names.set(v, name);
		return name;
	}
	nameType(v) {
		return `${this.name(v)}:${v.aval.toString()}`;
	}
};
/** A single statement / binding in a Jaxpr, in ANF form. */
var JaxprEqn = class {
	constructor(primitive, inputs, params, outBinders) {
		this.primitive = primitive;
		this.inputs = inputs;
		this.params = params;
		this.outBinders = outBinders;
	}
	pprint(usedVars, vp = new VarPrinter()) {
		const lhs = require_backend.PPrint.pp(this.outBinders.map((v) => !usedVars || usedVars.has(v) ? vp.nameType(v) : "_").join(" "));
		let rhs = require_backend.PPrint.pp(this.primitive);
		const paramsList = Object.entries(this.params).map(([k, v]) => require_backend.PPrint.pp(`${k}=${v}`));
		if (paramsList.length > 0) rhs = rhs.stack(require_backend.PPrint.pp(" [ ")).stack(require_backend.PPrint.prototype.concat(...paramsList)).stack(require_backend.PPrint.pp(" ] "));
		else rhs = rhs.stack(require_backend.PPrint.pp(" "));
		rhs = rhs.stack(require_backend.PPrint.pp(this.inputs.map((x) => x instanceof Var ? vp.name(x) : String(x.value)).join(" ")));
		return lhs.stack(require_backend.PPrint.pp(" = ")).stack(rhs);
	}
	toString() {
		return this.pprint().toString();
	}
};
/** Typed intermediate representation for traced computations. */
var Jaxpr = class Jaxpr {
	#hash;
	constructor(inBinders, eqns, outs) {
		this.inBinders = inBinders;
		this.eqns = eqns;
		this.outs = outs;
	}
	pprint() {
		const vp = new VarPrinter();
		const usedVars = new Set([...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter((x) => x instanceof Var));
		const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
		const eqns = require_backend.PPrint.prototype.concat(...this.eqns.map((e$1) => e$1.pprint(usedVars, vp)));
		const outs = this.outs.map((x) => x instanceof Var ? vp.name(x) : x.value).join(", ");
		return require_backend.PPrint.pp(`{ lambda ${inBinders} .`).concat((this.eqns.length ? require_backend.PPrint.pp("let ").stack(eqns).concat(require_backend.PPrint.pp(`in ( ${outs} ) }`)) : require_backend.PPrint.pp(`( ${outs} ) }`)).indent(2));
	}
	toString() {
		return this.pprint().toString();
	}
	/**
	* Gets a hash of this Jaxpr.
	*
	* Var identity is not considered in the hash, so two Jaxprs with the same
	* order of assignments and operators but different variable IDs will resolve
	* to the same hash (and toString representation).
	*/
	getHash() {
		if (this.#hash !== void 0) return this.#hash;
		const hasher = new require_backend.FpHash();
		const varIds = /* @__PURE__ */ new Map();
		const vi = (v) => {
			if (varIds.has(v)) return varIds.get(v);
			const id = varIds.size + 1;
			varIds.set(v, require_backend.FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
			return id;
		};
		hasher.update(this.inBinders.length);
		for (const x of this.inBinders) hasher.update(vi(x));
		hasher.update(this.eqns.length);
		for (const eqn of this.eqns) {
			hasher.update(eqn.primitive);
			hasher.update(eqn.inputs.length);
			for (const x of eqn.inputs) hasher.update(x instanceof Var ? vi(x) : x.value);
			hasher.update(JSON.stringify(eqn.params));
			hasher.update(eqn.outBinders.length);
			for (const x of eqn.outBinders) hasher.update(vi(x));
		}
		hasher.update(this.outs.length);
		for (const x of this.outs) hasher.update(x instanceof Var ? vi(x) : x.value);
		return this.#hash = hasher.value;
	}
	hash(state) {
		state.update(this.getHash());
	}
	/**
	* Produce a simplified Jaxpr with basic optimizations applied.
	*  - Trim away unused variables.
	*  - Fold away *1, *0, or +0 operations against literals.
	*  - Remove no-op movement operations.
	*/
	simplify() {
		const context = /* @__PURE__ */ new Map();
		const newEqns = [];
		for (const e$1 of this.eqns) {
			const inputs = e$1.inputs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
			const eqn = new JaxprEqn(e$1.primitive, inputs, e$1.params, e$1.outBinders);
			if (eqn.primitive === Primitive.Add) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 0)) context.set(c, b);
				else if (atomIsLit(b, 0)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.dtype === require_backend.DType.Bool ? Math.min(a.value + b.value, 1) : a.value + b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Neg) {
				const [a] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a)) context.set(c, new Lit(a.aval, -a.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Mul) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 1)) context.set(c, b);
				else if (atomIsLit(b, 1)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.value * b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Idiv) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(b, 1) && !require_backend.isFloatDtype(a.aval.dtype)) context.set(c, a);
				else newEqns.push(eqn);
			} else if ((eqn.primitive === Primitive.Broadcast || eqn.primitive === Primitive.Reshape) && require_backend.deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape) || eqn.primitive === Primitive.Transpose && eqn.params.perm.every((p, i) => p === i) || eqn.primitive === Primitive.Flip && eqn.params.axis.length === 0 || eqn.primitive === Primitive.Shrink && eqn.params.slice.every(([s, e$2], i) => s === 0 && e$2 === eqn.inputs[0].aval.shape[i]) || eqn.primitive === Primitive.Pad && eqn.params.width.every(([w0, w1]) => w0 === 0 && w1 === 0)) context.set(eqn.outBinders[0], eqn.inputs[0]);
			else newEqns.push(eqn);
		}
		const outs = this.outs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
		const usedVars = new Set(outs.filter((x) => x instanceof Var));
		const liveEqns = [];
		for (let i = newEqns.length - 1; i >= 0; i--) {
			const eqn = newEqns[i];
			if (eqn.outBinders.some((v) => usedVars.has(v))) {
				liveEqns.push(eqn);
				for (const v of eqn.inputs) if (v instanceof Var) usedVars.add(v);
			}
		}
		return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
	}
	/** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
	flatten() {
		if (!this.eqns.some((eqn) => eqn.primitive === Primitive.Jit)) return this;
		const newEqns = [];
		const varMap = /* @__PURE__ */ new Map();
		const varMapF = (x) => x instanceof Var ? varMap.get(x) ?? x : x;
		for (const eqn of this.eqns) if (eqn.primitive === Primitive.Jit) {
			const jaxpr = eqn.params.jaxpr.flatten();
			const translation = /* @__PURE__ */ new Map();
			const translationF = (x) => x instanceof Var ? translation.get(x) : x;
			for (const [v, x] of require_backend.zip(jaxpr.inBinders, eqn.inputs)) translation.set(v, varMapF(x));
			for (const ieqn of jaxpr.eqns) {
				const inputs = ieqn.inputs.map(translationF);
				const outBinders = [];
				for (const v of ieqn.outBinders) {
					const u = new Var(v.aval);
					outBinders.push(u);
					translation.set(v, u);
				}
				newEqns.push(new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders));
			}
			for (const [v, x] of require_backend.zip(eqn.outBinders, jaxpr.outs)) varMap.set(v, translationF(x));
		} else if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) newEqns.push(new JaxprEqn(eqn.primitive, eqn.inputs.map(varMapF), eqn.params, eqn.outBinders));
		else newEqns.push(eqn);
		const newOuts = this.outs.map(varMapF);
		return new Jaxpr(this.inBinders, newEqns, newOuts);
	}
};
var JaxprType = class {
	constructor(inTypes, outTypes) {
		this.inTypes = inTypes;
		this.outTypes = outTypes;
	}
	toString() {
		const inTypes = this.inTypes.map((aval) => aval.toString()).join(", ");
		const outTypes = this.outTypes.map((aval) => aval.toString()).join(", ");
		return `(${inTypes}) -> (${outTypes})`;
	}
};
function typecheckJaxpr(jaxpr) {
	const env = /* @__PURE__ */ new Set();
	for (const v of jaxpr.inBinders) {
		if (env.has(v)) throw new TypeError(`Duplicate variable binding: ${v}`);
		env.add(v);
	}
	for (const eqn of jaxpr.eqns) {
		const inTypes$1 = eqn.inputs.map((x) => typecheckAtom(env, x));
		const rule = abstractEvalRules[eqn.primitive];
		const outTypes$1 = rule(inTypes$1, eqn.params);
		for (const [outBinder, outType] of require_backend.zip(eqn.outBinders, outTypes$1)) {
			if (!outType.equals(outBinder.aval)) throw new TypeError(`Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`);
			if (env.has(outBinder)) throw new TypeError(`Duplicate variable binding: ${outBinder}`);
			env.add(outBinder);
		}
	}
	const inTypes = jaxpr.inBinders.map((v) => v.aval);
	const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
	return new JaxprType(inTypes, outTypes);
}
function typecheckAtom(env, x) {
	if (x instanceof Var) {
		if (!env.has(x)) throw new Error(`Unknown variable: ${x}`);
		return x.aval;
	} else if (x instanceof Lit) return x.aval;
	else throw new TypeError(`Invalid atom type: ${x}`);
}
/** Evaluate a Jaxpr on an array of inputs. */
function evalJaxpr(jaxpr, args) {
	const env = /* @__PURE__ */ new Map();
	const usageCount = /* @__PURE__ */ new Map();
	for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
	const remainingRefs = /* @__PURE__ */ new Map();
	const read = (x) => {
		if (x instanceof Var) {
			remainingRefs.set(x, (remainingRefs.get(x) ?? 0) - 1);
			return env.get(x);
		} else return array(x.value, { dtype: x.dtype });
	};
	const write = (v, val) => {
		if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
		let refCount = usageCount.get(v) ?? 0;
		if (refCount) {
			env.set(v, val);
			remainingRefs.set(v, refCount);
			while (refCount-- > 1) val.ref;
		} else val.dispose();
	};
	try {
		for (const [v, arg] of require_backend.zip(jaxpr.inBinders, args)) write(v, arg);
		for (const eqn of jaxpr.eqns) {
			const inVals = eqn.inputs.map(read);
			const outVals = bind(eqn.primitive, inVals, eqn.params);
			for (const [v, val] of require_backend.zip(eqn.outBinders, outVals)) write(v, val);
		}
		return jaxpr.outs.map(read);
	} catch (error) {
		for (let [v, refCount] of remainingRefs.entries()) if (refCount > 0) {
			const tracer = env.get(v);
			while (refCount--) tracer.dispose();
		}
		throw error;
	}
}
/** Convert a Jaxpr to a callable function by evaluating it. */
function jaxprAsFun(jaxpr) {
	return (...args) => evalJaxpr(jaxpr, args);
}
/** Jaxpr with a collection of associated, traced constants. */
var ClosedJaxpr = class ClosedJaxpr {
	constructor(jaxpr, consts) {
		this.jaxpr = jaxpr;
		this.consts = consts;
	}
	/** String representation of this Jaxpr. */
	toString() {
		return this.jaxpr.toString();
	}
	/** Apply a function to the underlying Jaxpr. */
	mapJaxpr(f) {
		return new ClosedJaxpr(f(this.jaxpr), this.consts);
	}
	/** Dispose of the constants in this Jaxpr. */
	dispose() {
		for (const c of this.consts) c.dispose();
	}
};
/** Tracer that records its operations to dynamically construct a Jaxpr. */
var JaxprTracer = class extends Tracer {
	#rc;
	constructor(trace$1, aval) {
		super(trace$1);
		this.aval = aval;
		this.#rc = 1;
	}
	toString() {
		return `JaxprTracer(${this.aval.toString()})`;
	}
	get ref() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc++;
		return this;
	}
	dispose() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc--;
	}
	trackLiftedConstant() {
		this.#rc++;
	}
};
/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
var JaxprTrace = class extends Trace {
	/** Register a Jaxpr argument with a given shape and return the tracer. */
	newArg(aval) {
		aval = ShapedArray.fromAval(aval);
		const tracer = this.builder.newTracer(this, aval);
		this.builder.addVar(tracer);
		return tracer;
	}
	/** Register a constant / literal in this Jaxpr. */
	getOrMakeConstTracer(val) {
		if (!(val instanceof Tracer)) val = pureArray(val);
		let tracer = this.builder.constTracers.get(val);
		if (tracer === void 0) {
			tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
			this.builder.addConst(tracer, val);
		} else {
			val.dispose();
			tracer.trackLiftedConstant();
		}
		return tracer;
	}
	pure = this.getOrMakeConstTracer;
	lift = this.getOrMakeConstTracer;
	processPrimitive(primitive, tracers, params) {
		const avalsIn = tracers.map((t) => {
			t.dispose();
			return t.aval;
		});
		const avalsOut = abstractEvalRules[primitive](avalsIn, params);
		const outTracers = avalsOut.map((aval) => this.builder.newTracer(this, aval));
		this.builder.addEqn(new JaxprEqn(primitive, tracers.map((t) => this.builder.getVar(t)), params, outTracers.map((t) => this.builder.addVar(t))));
		return outTracers;
	}
	get builder() {
		return this.main.globalData;
	}
};
/** Incrementally constructs a Jaxpr. */
var JaxprBuilder = class {
	eqns = [];
	tracerToVar = /* @__PURE__ */ new Map();
	constTracers = /* @__PURE__ */ new Map();
	constVals = /* @__PURE__ */ new Map();
	tracers = [];
	newTracer(trace$1, aval) {
		const tracer = new JaxprTracer(trace$1, aval);
		this.tracers.push(tracer);
		return tracer;
	}
	addEqn(eqn) {
		this.eqns.push(eqn);
	}
	addVar(tracer) {
		if (this.tracerToVar.has(tracer)) throw new Error(`Tracer was added as variable twice: ${tracer}`);
		const v = new Var(tracer.aval);
		this.tracerToVar.set(tracer, v);
		return v;
	}
	getVar(tracer) {
		const v = this.tracerToVar.get(tracer);
		if (v === void 0) throw new Error(`Could not find variable for tracer: ${tracer}`);
		return v;
	}
	addConst(tracer, val) {
		const v = this.addVar(tracer);
		this.constTracers.set(val, tracer);
		this.constVals.set(v, val);
		return v;
	}
	build(inTracers, outTracers) {
		const [constVars, consts] = require_backend.unzip2(this.constVals.entries());
		const t2v = this.getVar.bind(this);
		const inBinders = [...constVars, ...inTracers.map(t2v)];
		const outVars = outTracers.map(t2v);
		const jaxpr = new Jaxpr(inBinders, this.eqns, outVars);
		typecheckJaxpr(jaxpr);
		const cjaxpr = new ClosedJaxpr(jaxpr, consts);
		return _inlineLiterals(cjaxpr);
	}
};
function _inlineLiterals({ jaxpr, consts }) {
	const literals = /* @__PURE__ */ new Map();
	const constBinders = [];
	const newConsts = [];
	for (let i = 0; i < consts.length; i++) if (ndim$1(consts[i]) === 0 && consts[i] instanceof Array$1) {
		const ar = consts[i];
		literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
	} else {
		constBinders.push(jaxpr.inBinders[i]);
		newConsts.push(consts[i]);
	}
	const newEqns = jaxpr.eqns.map((eqn) => new JaxprEqn(eqn.primitive, eqn.inputs.map((x) => literals.get(x) ?? x), eqn.params, eqn.outBinders));
	const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
	const newJaxpr = new Jaxpr([...constBinders, ...jaxpr.inBinders.slice(consts.length)], newEqns, newOuts);
	typecheckJaxpr(newJaxpr);
	return new ClosedJaxpr(newJaxpr, newConsts);
}
function binopAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("binopAbstractEval expects ShapedArray inputs");
	return [promoteAvals(x, y)];
}
function compareAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("compareAbstractEval expects ShapedArray inputs");
	const aval = promoteAvals(x, y);
	return [new ShapedArray(aval.shape, require_backend.DType.Bool, false)];
}
function vectorizedUnopAbstractEval([x]) {
	return [ShapedArray.fromAval(x)];
}
const abstractEvalRules = {
	[Primitive.Add]: binopAbstractEval,
	[Primitive.Mul]: binopAbstractEval,
	[Primitive.Idiv]: binopAbstractEval,
	[Primitive.Mod]: binopAbstractEval,
	[Primitive.Min]: binopAbstractEval,
	[Primitive.Max]: binopAbstractEval,
	[Primitive.Neg]: vectorizedUnopAbstractEval,
	[Primitive.Reciprocal]: vectorizedUnopAbstractEval,
	[Primitive.Floor]: vectorizedUnopAbstractEval,
	[Primitive.Ceil]: vectorizedUnopAbstractEval,
	[Primitive.StopGradient]: vectorizedUnopAbstractEval,
	[Primitive.Cast]([x], { dtype }) {
		return [new ShapedArray(x.shape, dtype, false)];
	},
	[Primitive.Bitcast]([x], { dtype }) {
		if (x.dtype === require_backend.DType.Bool || dtype === require_backend.DType.Bool) throw new TypeError("Bitcast to/from bool is not allowed");
		if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
		return [new ShapedArray(x.shape, dtype, false)];
	},
	[Primitive.Sin]: vectorizedUnopAbstractEval,
	[Primitive.Cos]: vectorizedUnopAbstractEval,
	[Primitive.Asin]: vectorizedUnopAbstractEval,
	[Primitive.Atan]: vectorizedUnopAbstractEval,
	[Primitive.Exp]: vectorizedUnopAbstractEval,
	[Primitive.Log]: vectorizedUnopAbstractEval,
	[Primitive.Erf]: vectorizedUnopAbstractEval,
	[Primitive.Erfc]: vectorizedUnopAbstractEval,
	[Primitive.Sqrt]: vectorizedUnopAbstractEval,
	[Primitive.Reduce]([x], { axis }) {
		const axisSet = new Set(axis);
		const newShape = x.shape.filter((_, i) => !axisSet.has(i));
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Pool]([x], { window, strides }) {
		const shape$1 = checkPoolShape(x.shape, window, strides);
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.PoolTranspose]([x], { inShape, window, strides }) {
		const shape$1 = checkPoolShape(inShape, window, strides);
		if (!require_backend.deepEqual(shape$1, x.shape)) throw new TypeError(`PoolTranspose shape mismatch: expected ${JSON.stringify(shape$1)}, got ${JSON.stringify(x.shape)}`);
		return [new ShapedArray(inShape, x.dtype, x.weakType)];
	},
	[Primitive.Dot]([x, y]) {
		if (x.ndim === 0 && y.ndim === 0) throw new TypeError("Dot requires at least 1D inputs");
		const { shape: shape$1, dtype, weakType } = promoteAvals(x, y);
		shape$1.splice(-1, 1);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Conv]([lhs, rhs], params) {
		const { dtype, weakType } = promoteAvals(lhs.scalar(), rhs.scalar());
		const shape$1 = checkConvShape(lhs.shape, rhs.shape, params);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Compare]: compareAbstractEval,
	[Primitive.Where]([cond, x, y]) {
		if (cond.dtype !== require_backend.DType.Bool) throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
		const xy = promoteAvals(x, y);
		const shape$1 = require_backend.generalBroadcast(cond.shape, xy.shape);
		return [new ShapedArray(shape$1, xy.dtype, xy.weakType)];
	},
	[Primitive.Concatenate](xs, { axis }) {
		if (xs.length === 0) throw new TypeError("Concatenate requires at least one input");
		for (const x of xs) if (x.ndim !== xs[0].ndim || !x.shape.every((s, i) => i === axis || s === xs[0].shape[i])) throw new TypeError(`Concatenate: inputs ${xs[0]} and ${x} must match shapes except on axis ${axis}`);
		const shape$1 = xs[0].shape.slice();
		shape$1[axis] = xs.reduce((sum$1, x) => sum$1 + x.shape[axis], 0);
		const { dtype, weakType } = xs.map((x) => x.scalar()).reduce(promoteAvals);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Split]([x], { axis, sizes }) {
		const totalSize = sizes.reduce((a, b) => a + b, 0);
		if (x.shape[axis] !== totalSize) throw new TypeError(`Split: sizes ${sizes} do not sum to dimension ${x.shape[axis]} on axis ${axis}`);
		return sizes.map((size$1) => {
			return new ShapedArray(x.shape.toSpliced(axis, 1, size$1), x.dtype, x.weakType);
		});
	},
	[Primitive.RandomBits]([k0, k1], { shape: shape$1 }) {
		if (k0.dtype !== require_backend.DType.Uint32 || k1.dtype !== require_backend.DType.Uint32) throw new TypeError(`RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`);
		if (!require_backend.deepEqual(k0.shape, k1.shape)) throw new TypeError(`RandomBits: Keys have different shapes ${k0.shape} and ${k1.shape}`);
		if (!require_backend.deepEqual(shape$1.slice(0, k0.ndim), k0.shape)) throw new TypeError(`RandomBits: generated shape ${shape$1} must match key shape ${k0.shape}`);
		return [new ShapedArray(shape$1, require_backend.DType.Uint32, false)];
	},
	[Primitive.Gather]([x, ...indices], { axis, outDim }) {
		for (const a of indices) if (a.dtype !== require_backend.DType.Int32 && a.dtype !== require_backend.DType.Uint32) throw new TypeError(`Gather indices must be Int32 or Uint32, got ${a.dtype}`);
		if (axis.length !== indices.length) throw new TypeError(`Gather: ${axis} axes but ${indices.length} indices`);
		if (indices.length === 0) throw new TypeError("Gather must have 1+ indices with same shape");
		if (axis.some((a) => a < 0 || a >= x.shape.length)) throw new TypeError("Gather axis out of bounds");
		if (outDim < 0 || outDim > x.shape.length - axis.length) throw new TypeError("Gather outDim out of bounds");
		const axisSet = new Set(axis);
		if (axisSet.size !== axis.length) throw new TypeError("Gather axes are not unique");
		const gatherShape = indices.reduce((shape$1, a) => require_backend.generalBroadcast(shape$1, a.shape), []);
		const newShape = x.shape.filter((_, i) => !axisSet.has(i));
		newShape.splice(outDim, 0, ...gatherShape);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Transpose]([x], { perm }) {
		return [new ShapedArray(perm.map((i) => x.shape[i]), x.dtype, x.weakType)];
	},
	[Primitive.Broadcast]([x], { shape: shape$1 }) {
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.Reshape]([x], { shape: shape$1 }) {
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.Flip]([x], _) {
		return [ShapedArray.fromAval(x)];
	},
	[Primitive.Shrink]([x], { slice }) {
		const newShape = slice.map((s) => s[1] - s[0]);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Pad]([x], { width }) {
		const newShape = x.shape.map((dim, i) => dim + width[i][0] + width[i][1]);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Sort]([x]) {
		if (x.ndim === 0) throw new TypeError("sort: requires at least 1D input");
		return [ShapedArray.fromAval(x)];
	},
	[Primitive.Argsort]([x]) {
		if (x.ndim === 0) throw new TypeError("argsort: requires at least 1D input");
		return [ShapedArray.fromAval(x), new ShapedArray(x.shape, require_backend.DType.Int32, false)];
	},
	[Primitive.TriangularSolve]([a, b]) {
		if (a.ndim < 2) throw new TypeError(`triangular_solve: a must be at least 2D, got ${a}`);
		if (b.ndim < 2) throw new TypeError(`triangular_solve: b must be at least 2D, got ${b}`);
		const [m, n] = a.shape.slice(-2);
		const [_batch, q] = b.shape.slice(-2);
		if (!require_backend.deepEqual(a.shape.slice(0, -2), b.shape.slice(0, -2)) || a.dtype !== b.dtype || m !== n || n !== q) throw new TypeError(`triangular_solve: mismatch ${a} vs ${b}`);
		return [new ShapedArray(b.shape, b.dtype, a.weakType && b.weakType)];
	},
	[Primitive.Cholesky]([a]) {
		if (a.ndim < 2) throw new TypeError(`cholesky: requires at least 2D input, got ${a}`);
		if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`cholesky: must be square, got ${a}`);
		return [ShapedArray.fromAval(a)];
	},
	[Primitive.LU]([a]) {
		if (a.ndim < 2) throw new TypeError(`lu: requires at least 2D input, got ${a}`);
		const batch = a.shape.slice(0, -2);
		const [m, n] = a.shape.slice(-2);
		return [
			ShapedArray.fromAval(a),
			new ShapedArray([...batch, Math.min(m, n)], require_backend.DType.Int32, false),
			new ShapedArray([...batch, m], require_backend.DType.Int32, false)
		];
	},
	[Primitive.Jit](args, { jaxpr }) {
		const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
		if (args.length !== inTypes.length) throw new TypeError(`jit expected ${inTypes.length} arguments, got ${args.length}`);
		for (let i = 0; i < inTypes.length; i++) if (!args[i].equals(inTypes[i])) throw new TypeError(`jit argument ${i} has type ${args[i]}, expected ${inTypes[i]}`);
		return outTypes;
	},
	[Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse: _ }) {
		const numX = args.length - numConsts - numCarry;
		const { outTypes } = typecheckJaxpr(jaxpr);
		if (jaxpr.inBinders.length !== numConsts + numCarry + numX) throw new TypeError(`Scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`);
		const carryOutTypes = outTypes.slice(0, numCarry);
		const ySliceTypes = outTypes.slice(numCarry);
		const yTypes = ySliceTypes.map((t) => {
			return new ShapedArray([length, ...t.shape], t.dtype, t.weakType);
		});
		return [...carryOutTypes, ...yTypes];
	}
};
function splitIdx(values, argnums) {
	const a = [];
	const b = [];
	for (let i = 0; i < values.length; i++) if (argnums.has(i)) a.push(values[i]);
	else b.push(values[i]);
	return [a, b];
}
function joinIdx(n, a, b, argnums) {
	const result = [];
	let ai = 0;
	let bi = 0;
	for (let i = 0; i < n; i++) if (argnums.has(i)) result.push(a[ai++]);
	else result.push(b[bi++]);
	return result;
}
function makeJaxpr$1(f, opts) {
	return (...argsIn) => {
		try {
			var _usingCtx$1 = (0, import_usingCtx$2.default)();
			const staticArgnums = new Set(opts?.staticArgnums ?? []);
			const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);
			const [avalsIn, inTree] = flatten(shapedArgs);
			const [fFlat, outTree] = flattenFun((...dynamicArgs) => {
				return f(...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums));
			}, inTree);
			const builder = new JaxprBuilder();
			const main = _usingCtx$1.u(newMain(JaxprTrace, builder));
			_usingCtx$1.u(newDynamic(main));
			const trace$1 = new JaxprTrace(main);
			const tracersIn = avalsIn.map((aval) => trace$1.newArg(typeof aval === "object" ? aval : pureArray(aval)));
			const outs = fFlat(...tracersIn);
			const tracersOut = outs.map((out) => fullRaise(trace$1, out));
			const jaxpr = builder.build(tracersIn, tracersOut);
			if (outTree.value === void 0) throw new Error("outTree was not set in makeJaxpr");
			return {
				jaxpr: jaxpr.mapJaxpr((j) => j.simplify()),
				treedef: outTree.value
			};
		} catch (_) {
			_usingCtx$1.e = _;
		} finally {
			_usingCtx$1.d();
		}
	};
}
function jit$1(f, opts) {
	const cache = /* @__PURE__ */ new Map();
	const staticArgnums = new Set(opts?.staticArgnums ?? []);
	const result = ((...args) => {
		const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);
		const [argsFlat, inTree] = flatten(dynamicArgs);
		const avalsInFlat = argsFlat.map((x) => ShapedArray.fromAval(getAval(x)));
		const avalsIn = unflatten(inTree, avalsInFlat);
		const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
		const { jaxpr, treedef: outTree } = require_backend.runWithCache(cache, jaxprArgs, () => makeJaxpr$1(f, opts)(...jaxprArgs));
		const outs = bind(Primitive.Jit, [...jaxpr.consts.map((c) => c.ref), ...argsFlat], {
			name: f.name || "closure",
			jaxpr: jaxpr.jaxpr,
			numConsts: jaxpr.consts.length
		});
		return unflatten(outTree, outs);
	});
	result.dispose = () => {
		for (const { jaxpr } of cache.values()) jaxpr.dispose();
	};
	return result;
}

//#endregion
//#region src/frontend/jit.ts
/**
* Check if a chosen scan path satisfies the requirePath constraint.
* Returns an error message if the path is not allowed, or null if OK.
*/
function checkRequiredPath(chosenPath, requirePath) {
	if (!requirePath) return null;
	const allowedPaths = Array.isArray(requirePath) ? requirePath : [requirePath];
	if (!allowedPaths.includes(chosenPath)) return `Scan requirePath constraint not satisfied: got "${chosenPath}" but required one of [${allowedPaths.map((p) => `"${p}"`).join(", ")}]`;
	return null;
}
/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
var JitProgram = class {
	constructor(backend, steps, inputs, outputs) {
		this.backend = backend;
		this.steps = steps;
		this.inputs = inputs;
		this.outputs = outputs;
	}
	pprint() {
		const steps = this.steps.map((step) => {
			switch (step.type) {
				case "execute": {
					const inputsNice = step.inputs.map((id, i) => `${i}: %${id}`).join(", ");
					const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
					const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
					if (step.source instanceof require_backend.Kernel) return require_backend.PPrint.pp(`${executeText}, kernel`).concat(step.source.pprint().indent(2));
					else if (step.source instanceof require_backend.Routine) return require_backend.PPrint.pp(`${executeText}, routine ${step.source.name}`);
					else {
						step.source;
						return require_backend.PPrint.pp(executeText);
					}
				}
				case "malloc": return require_backend.PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
				case "incref": return require_backend.PPrint.pp(`incref ${step.input}`);
				case "free": return require_backend.PPrint.pp(`free ${step.input}`);
				case "scan": return require_backend.PPrint.pp(`scan length=${step.length} numCarry=${step.numCarry} numConsts=${step.numConsts}`).concat(require_backend.PPrint.pp(`  consts=[${step.consts.join(", ")}] initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`)).concat(require_backend.PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`)).concat(require_backend.PPrint.pp("  body=").concat(require_backend.PPrint.pp(step.bodyJaxpr.toString()).indent(4)));
				case "native-scan": return require_backend.PPrint.pp(`native-scan length=${step.length} numCarry=${step.numCarry}`).concat(require_backend.PPrint.pp(`  initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`)).concat(require_backend.PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`));
				case "batched-scan": return require_backend.PPrint.pp(`batched-scan length=${step.length} numCarry=${step.numCarry} numConsts=${step.numConsts}`).concat(require_backend.PPrint.pp(`  initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`)).concat(require_backend.PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`));
			}
		});
		const display = require_backend.PPrint.prototype.concat(require_backend.PPrint.pp(`device = ${this.backend.type}`), require_backend.PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"), require_backend.PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"), require_backend.PPrint.pp("steps ="), require_backend.PPrint.prototype.concat(...steps).indent(2));
		return require_backend.PPrint.pp("{ ").stack(display.stack(require_backend.PPrint.pp(" }")));
	}
	toString() {
		return this.pprint().toString();
	}
	/** Execute the JitProgram with the given inputs.
	* @param scanRunner - Optional callback to run scan steps. Required if program contains scan steps.
	*/
	execute(inputs, scanRunner) {
		const scope = /* @__PURE__ */ new Map();
		if (inputs.length !== this.inputs.length) throw new TypeError(`Expected ${this.inputs.length} inputs, got ${inputs.length}`);
		for (const [i, id] of this.inputs.entries()) scope.set(id, inputs[i]);
		const pending = [];
		for (const step of this.steps) switch (step.type) {
			case "execute": {
				const inputs$1 = step.inputs.map((id) => scope.get(id));
				const outputs = step.outputs.map((id) => scope.get(id));
				if (inputs$1.some((s) => s === void 0) || outputs.some((s) => s === void 0)) throw new Error(`internal: JitProgram scope undefined`);
				pending.push(new PendingExecute(this.backend, step.source, inputs$1, outputs));
				break;
			}
			case "malloc": {
				const slot = this.backend.malloc(step.size);
				scope.set(step.output, slot);
				break;
			}
			case "incref": {
				const slot = scope.get(step.input);
				this.backend.incRef(slot);
				break;
			}
			case "free": {
				const slot = scope.get(step.input);
				this.backend.decRef(slot);
				scope.delete(step.input);
				break;
			}
			case "scan": {
				if (!scanRunner) throw new Error("internal: scan step requires scanRunner callback");
				for (const p of pending) {
					p.prepareSync();
					p.submit();
				}
				pending.length = 0;
				const constSlots = step.consts.map((id) => scope.get(id));
				const initCarrySlots = step.initCarry.map((id) => scope.get(id));
				const xsSlots = step.xs.map((id) => scope.get(id));
				const outputSlots = step.outputs.map((id) => scope.get(id));
				if (require_backend.DEBUG >= 2) {
					console.log(`[jit.scan] step.xs=${step.xs}, xsSlots=${xsSlots}`);
					console.log(`[jit.scan] step.xsAvals=${JSON.stringify(step.xsAvals?.map((a) => ({
						shape: a.shape,
						dtype: a.dtype
					})))}`);
					for (let i = 0; i < xsSlots.length; i++) {
						const data = this.backend.readSync(xsSlots[i]);
						console.log(`[jit.scan] xs[${i}] data:`, new Float32Array(data.buffer));
					}
				}
				if (require_backend.DEBUG >= 2) console.log(`[jit.scan] Before scanRunner: outputSlots=${outputSlots}, step.outputs=${step.outputs}`);
				const result = scanRunner(step.bodyProgram, this.backend, step.bodyJaxpr, step.length, step.numCarry, step.numConsts, step.numX, step.numY, step.reverse, constSlots, initCarrySlots, xsSlots, step.xsAvals, outputSlots);
				if (require_backend.DEBUG >= 2) console.log(`[jit.scan] After scanRunner: result.outputs=${result.outputs}`);
				for (let i = 0; i < step.outputs.length; i++) scope.set(step.outputs[i], result.outputs[i]);
				if (require_backend.DEBUG >= 2) console.log(`[jit.scan] After scope.set: scope.get(${step.outputs[0]})=${scope.get(step.outputs[0])}`);
				pending.push(...result.pending);
				break;
			}
			case "native-scan": {
				for (const p of pending) {
					p.prepareSync();
					p.submit();
				}
				pending.length = 0;
				const constSlots = step.consts.map((id) => scope.get(id));
				const initCarrySlots = step.initCarry.map((id) => scope.get(id));
				const xsSlots = step.xs.map((id) => scope.get(id));
				const outputSlots = step.outputs.map((id) => scope.get(id));
				const carryOutSlots = outputSlots.slice(0, step.numCarry);
				const ysStackedSlots = outputSlots.slice(step.numCarry);
				const backend = this.backend;
				if (step.generalParams) if (typeof backend.dispatchNativeScanGeneral === "function") backend.dispatchNativeScanGeneral(step.executable, step.generalParams, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
				else throw new Error("internal: general native-scan requires backend.dispatchNativeScanGeneral");
				else if (typeof backend.dispatchNativeScan === "function") backend.dispatchNativeScan(step.executable, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
				else throw new Error("internal: native-scan requires backend.dispatchNativeScan");
				break;
			}
			case "batched-scan": {
				for (const p of pending) {
					p.prepareSync();
					p.submit();
				}
				pending.length = 0;
				const constSlots = step.consts.map((id) => scope.get(id));
				const initCarrySlots = step.initCarry.map((id) => scope.get(id));
				const xsSlots = step.xs.map((id) => scope.get(id));
				const outputSlots = step.outputs.map((id) => scope.get(id));
				const carryOutSlots = outputSlots.slice(0, step.numCarry);
				const ysStackedSlots = outputSlots.slice(step.numCarry);
				const backend = this.backend;
				if (typeof backend.dispatchBatchedScan === "function") backend.dispatchBatchedScan(step.batchedParams, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
				else throw new Error("internal: batched-scan requires backend.dispatchBatchedScan");
				break;
			}
			default:
		}
		return {
			outputs: this.outputs.map((id) => scope.get(id)),
			pending
		};
	}
};
var JitProgramBuilder = class {
	backend;
	#nextId;
	steps;
	constructor(backend, nargs) {
		this.backend = backend;
		this.#nextId = nargs;
		this.steps = [];
	}
	pushLit(lit) {
		const kernel = new require_backend.Kernel(0, lit.aval.size, require_backend.AluExp.const(lit.dtype, lit.value));
		return this.pushKernel(kernel, []);
	}
	pushBuffer(size$1) {
		const id = this.#nextId++;
		this.steps.push({
			type: "malloc",
			size: size$1,
			output: id
		});
		return id;
	}
	pushKernel(kernel, inputs) {
		const id = this.pushBuffer(kernel.bytes);
		this.steps.push({
			type: "execute",
			source: kernel,
			inputs,
			outputs: [id]
		});
		return id;
	}
	pushRoutine(routine, inputs, outputs) {
		this.steps.push({
			type: "execute",
			source: routine,
			inputs,
			outputs
		});
	}
	pushIncref(id) {
		this.steps.push({
			type: "incref",
			input: id
		});
	}
	insertFreeSteps(outputIds) {
		const ids = this.steps.filter((s) => s.type === "malloc").map((s) => s.output);
		for (const id of ids) {
			if (outputIds.includes(id)) continue;
			const lastUsage = this.steps.findLastIndex((s) => s.type === "execute" && (s.outputs.includes(id) || s.inputs.includes(id)) || s.type === "malloc" && s.output === id || s.type === "scan" && (s.consts.includes(id) || s.initCarry.includes(id) || s.xs.includes(id) || s.outputs.includes(id)) || s.type === "native-scan" && (s.consts.includes(id) || s.initCarry.includes(id) || s.xs.includes(id) || s.outputs.includes(id)) || s.type === "batched-scan" && (s.consts.includes(id) || s.initCarry.includes(id) || s.xs.includes(id) || s.outputs.includes(id)));
			this.steps.splice(lastUsage + 1, 0, {
				type: "free",
				input: id
			});
		}
	}
	pushFree(id) {
		this.steps.push({
			type: "free",
			input: id
		});
	}
};
const jitCompileCache = /* @__PURE__ */ new Map();
function jitCompile(backend, jaxpr) {
	const cacheKey = backend.type + "," + require_backend.FpHash.hash(jaxpr);
	const cached = jitCompileCache.get(cacheKey);
	if (cached) return cached;
	jaxpr = jaxpr.flatten().simplify();
	const nargs = jaxpr.inBinders.length;
	const builder = new JitProgramBuilder(backend, nargs);
	const blackNodes = splitGraphDataflow(backend, jaxpr);
	const ctx = /* @__PURE__ */ new Map();
	for (let i = 0; i < nargs; i++) {
		const v = jaxpr.inBinders[i];
		ctx.set(v, {
			type: "imm",
			arg: i
		});
	}
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (routinePrimitives.has(eqn.primitive)) {
			const routine = new require_backend.Routine(routinePrimitives.get(eqn.primitive), {
				inputShapes: eqn.inputs.map((x) => x.aval.shape),
				inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
				outputShapes: eqn.outBinders.map((x) => x.aval.shape),
				outputDtypes: eqn.outBinders.map((x) => x.aval.dtype)
			}, eqn.params);
			const inputs = [];
			for (const input of eqn.inputs) if (input instanceof Var) {
				const jv = ctx.get(input);
				if (jv.type !== "imm") throw new Error(`jit: routine primitive ${eqn.primitive} input is not imm`);
				inputs.push(jv.arg);
			} else if (input instanceof Lit) inputs.push(builder.pushLit(input));
			const outputs = [];
			for (const outVar of eqn.outBinders) {
				const outId = builder.pushBuffer(outVar.aval.size * require_backend.byteWidth(outVar.aval.dtype));
				outputs.push(outId);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			}
			builder.pushRoutine(routine, inputs, outputs);
			continue;
		}
		if (eqn.primitive === Primitive.Scan) {
			const params = eqn.params;
			const { jaxpr: bodyJaxpr, numCarry, numConsts, length, reverse, requirePath } = params;
			const numX = bodyJaxpr.inBinders.length - numConsts - numCarry;
			const numY = bodyJaxpr.outs.length - numCarry;
			const inputs = [];
			for (const input of eqn.inputs) if (input instanceof Var) {
				const jv = ctx.get(input);
				if (jv.type !== "imm") throw new Error(`jit: scan primitive input is not imm`);
				inputs.push(jv.arg);
			} else if (input instanceof Lit) inputs.push(builder.pushLit(input));
			const constsIds = inputs.slice(0, numConsts);
			const initCarryIds = inputs.slice(numConsts, numConsts + numCarry);
			const xsIds = inputs.slice(numConsts + numCarry);
			const xsAvals = [];
			const xsInputs = eqn.inputs.slice(numConsts + numCarry);
			for (const input of xsInputs) if (input instanceof Var) xsAvals.push(input.aval);
			else if (input instanceof Lit) xsAvals.push(input.aval);
			const outputs = [];
			for (const outVar of eqn.outBinders) {
				const outId = builder.pushBuffer(outVar.aval.size * require_backend.byteWidth(outVar.aval.dtype));
				outputs.push(outId);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			}
			const bodyProgram = jitCompile(backend, bodyJaxpr);
			let nativeScanExe = tryPrepareNativeScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, eqn, reverse);
			if (!nativeScanExe) {
				if (require_backend.DEBUG >= 2) console.log(`[scan] Single-kernel native scan failed, trying multi-kernel...`);
				nativeScanExe = tryPrepareNativeScanMulti(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse);
			}
			let generalScanResult = null;
			if (!nativeScanExe) {
				if (require_backend.DEBUG >= 2) console.log(`[scan] Multi-kernel native scan failed, trying general native scan...`);
				generalScanResult = tryPrepareNativeScanGeneral(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse);
				if (generalScanResult) nativeScanExe = generalScanResult.executable;
			}
			if (nativeScanExe) {
				const pathError$1 = checkRequiredPath("fused", requirePath);
				if (pathError$1) throw new Error(pathError$1);
				require_backend.reportScanPath("fused", backend.type, {
					numConsts,
					numCarry,
					length
				});
				builder.steps.push({
					type: "native-scan",
					executable: nativeScanExe,
					length,
					numCarry,
					numConsts,
					numY,
					reverse,
					consts: constsIds,
					initCarry: initCarryIds,
					xs: xsIds,
					outputs,
					generalParams: generalScanResult?.params
				});
				continue;
			}
			const batchedParams = tryPrepareBatchedScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, eqn, reverse);
			if (batchedParams) {
				const pathError$1 = checkRequiredPath("fused", requirePath);
				if (pathError$1) throw new Error(pathError$1);
				require_backend.reportScanPath("fused", backend.type, {
					numConsts,
					numCarry,
					length
				});
				builder.steps.push({
					type: "batched-scan",
					batchedParams,
					length,
					numCarry,
					numConsts,
					numX,
					numY,
					reverse,
					consts: constsIds,
					initCarry: initCarryIds,
					xs: xsIds,
					outputs
				});
				continue;
			}
			const pathError = checkRequiredPath("fallback", requirePath);
			if (pathError) throw new Error(pathError);
			require_backend.reportScanPath("fallback", backend.type, {
				numConsts,
				numCarry,
				length
			});
			builder.steps.push({
				type: "scan",
				bodyProgram,
				bodyJaxpr,
				length,
				numCarry,
				numConsts,
				numX,
				numY,
				reverse,
				consts: constsIds,
				initCarry: initCarryIds,
				xs: xsIds,
				xsAvals,
				outputs
			});
			continue;
		}
		const inputExps = [];
		const inputAvals = [];
		const inputArgs = [];
		let inputReduction = null;
		const addArgs = (args) => {
			const newGids = [];
			for (const jitId of args) {
				let newGid = inputArgs.indexOf(jitId);
				if (newGid === -1) {
					newGid = inputArgs.length;
					inputArgs.push(jitId);
				}
				newGids.push(newGid);
			}
			return newGids;
		};
		for (const input of eqn.inputs) if (input instanceof Var) {
			const jv = ctx.get(input);
			if (jv.type === "exp") {
				const newGids = addArgs(jv.args);
				inputExps.push(jv.exp.reindexGids(newGids));
			} else if (jv.type === "imm") {
				const [gid] = addArgs([jv.arg]);
				const st = require_backend.ShapeTracker.fromShape(input.aval.shape);
				const indices = require_backend.unravelAlu(st.shape, require_backend.AluVar.gidx);
				inputExps.push(require_backend.AluExp.globalView(input.aval.dtype, gid, st, indices));
			} else if (jv.type === "red") {
				if (inputReduction) throw new Error("jit: unexpected, multiple red inputs");
				const newGids = addArgs(jv.args);
				inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
				inputReduction = jv;
			}
			inputAvals.push(input.aval);
		} else if (input instanceof Lit) {
			inputExps.push(require_backend.AluExp.const(input.dtype, input.value));
			inputAvals.push(input.aval);
		} else throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
		const rule = jitRules[eqn.primitive];
		if (!rule) throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
		let exp$2;
		let reduction;
		if (inputReduction) {
			const jv = inputReduction;
			const newEpilogue = rule(inputExps, inputAvals, eqn.params).exp[0];
			exp$2 = [jv.exp.reindexGids(addArgs(jv.args))];
			reduction = new require_backend.Reduction(jv.reduction.dtype, jv.reduction.op, jv.reduction.size, newEpilogue);
		} else {
			const ruleOutput = rule(inputExps, inputAvals, eqn.params);
			exp$2 = ruleOutput.exp;
			reduction = ruleOutput.reduction;
		}
		for (let i$1 = 0; i$1 < eqn.outBinders.length; i$1++) {
			const outVar = eqn.outBinders[i$1];
			if (blackNodes.has(outVar)) {
				const nargs$1 = inputArgs.length;
				const size$1 = outVar.aval.size;
				const kernel = new require_backend.Kernel(nargs$1, size$1, exp$2[i$1], reduction);
				const outId = builder.pushKernel(kernel, inputArgs);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			} else if (reduction) ctx.set(outVar, {
				type: "red",
				exp: exp$2[i$1],
				reduction,
				args: inputArgs
			});
			else ctx.set(outVar, {
				type: "exp",
				exp: exp$2[i$1],
				args: inputArgs
			});
		}
	}
	const outputIds = [];
	for (const out of jaxpr.outs) if (out instanceof Var) {
		const jitValue = ctx.get(out);
		if (jitValue.type !== "imm") throw new Error("internal: Expected imm, since outs are black nodes");
		outputIds.push(jitValue.arg);
	} else if (out instanceof Lit) outputIds.push(builder.pushLit(out));
	const outputNeedsRef = new Set(require_backend.range(nargs));
	for (const outputId of outputIds) if (outputNeedsRef.has(outputId)) builder.pushIncref(outputId);
	else outputNeedsRef.add(outputId);
	builder.insertFreeSteps(outputIds);
	const jp = new JitProgram(backend, builder.steps, require_backend.range(0, nargs), outputIds);
	jitCompileCache.set(cacheKey, jp);
	return jp;
}
function reshapeViews(exp$2, mapping, reduceAxis = false) {
	return exp$2.rewrite((exp$3) => {
		if (exp$3.op === require_backend.AluOp.GlobalView) {
			const [gid, st] = exp$3.arg;
			const newSt = mapping(st);
			if (newSt) {
				const indices = reduceAxis ? require_backend.unravelAlu(newSt.shape.slice(0, -1), require_backend.AluVar.gidx).concat(require_backend.AluVar.ridx) : require_backend.unravelAlu(newSt.shape, require_backend.AluVar.gidx);
				return require_backend.AluExp.globalView(exp$3.dtype, gid, newSt, indices);
			}
		} else if (exp$3.op === require_backend.AluOp.GlobalIndex) throw new Error("internal: reshapeViews() called with GlobalIndex op");
	});
}
function broadcastedJit(fn, opts) {
	return (exps, avals, params) => {
		let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);
		const skipCastIdx = opts?.skipCastIdx ?? [];
		if (skipCastIdx.length) newDtype = avals.filter((_, i) => !skipCastIdx.includes(i)).reduce(promoteAvals).dtype;
		exps = exps.map((exp$2, i) => {
			exp$2 = reshapeViews(exp$2, (st) => {
				if (!require_backend.deepEqual(st.shape, newShape)) return st.broadcast(newShape, require_backend.range(newShape.length - st.shape.length));
			});
			if (exp$2.dtype !== newDtype && !skipCastIdx.includes(i)) exp$2 = require_backend.AluExp.cast(newDtype, exp$2);
			return exp$2;
		});
		return { exp: [fn(exps, params)] };
	};
}
function unopJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [fn(a, params)] };
	};
}
function reshapeJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [reshapeViews(a, (st) => fn(st, params))] };
	};
}
function routineNoJit() {
	return () => {
		throw new Error("jit: rule is not implemented for routines");
	};
}
const jitRules = {
	[Primitive.Add]: broadcastedJit(([a, b]) => require_backend.AluExp.add(a, b)),
	[Primitive.Mul]: broadcastedJit(([a, b]) => require_backend.AluExp.mul(a, b)),
	[Primitive.Idiv]: broadcastedJit(([a, b]) => require_backend.AluExp.idiv(a, b)),
	[Primitive.Mod]: broadcastedJit(([a, b]) => require_backend.AluExp.mod(a, b)),
	[Primitive.Min]: broadcastedJit(([a, b]) => require_backend.AluExp.min(a, b)),
	[Primitive.Max]: broadcastedJit(([a, b]) => require_backend.AluExp.max(a, b)),
	[Primitive.Neg]: unopJit((a) => require_backend.AluExp.sub(require_backend.AluExp.const(a.dtype, 0), a)),
	[Primitive.Reciprocal]: unopJit(require_backend.AluExp.reciprocal),
	[Primitive.Floor]: unopJit(require_backend.AluExp.floor),
	[Primitive.Ceil]: unopJit(require_backend.AluExp.ceil),
	[Primitive.StopGradient]: unopJit((a) => a),
	[Primitive.Cast]: unopJit((a, { dtype }) => require_backend.AluExp.cast(dtype, a)),
	[Primitive.Bitcast]: unopJit((a, { dtype }) => require_backend.AluExp.bitcast(dtype, a)),
	[Primitive.Sin]: unopJit(require_backend.AluExp.sin),
	[Primitive.Cos]: unopJit(require_backend.AluExp.cos),
	[Primitive.Asin]: unopJit(require_backend.AluExp.asin),
	[Primitive.Atan]: unopJit(require_backend.AluExp.atan),
	[Primitive.Exp]: unopJit(require_backend.AluExp.exp),
	[Primitive.Log]: unopJit(require_backend.AluExp.log),
	[Primitive.Erf]: unopJit(require_backend.AluExp.erf),
	[Primitive.Erfc]: unopJit(require_backend.AluExp.erfc),
	[Primitive.Sqrt]: unopJit(require_backend.AluExp.sqrt),
	[Primitive.Reduce]([a], [as], { op, axis }) {
		const keptAxes = [];
		const shiftedAxes = [];
		const newShape = [];
		for (let i = 0; i < as.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(as.shape[i]);
		}
		const reductionSize = require_backend.prod(shiftedAxes.map((ax) => as.shape[ax]));
		newShape.push(reductionSize);
		const perm = keptAxes.concat(shiftedAxes);
		a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
		const reduction = new require_backend.Reduction(a.dtype, op, reductionSize);
		return {
			exp: [a],
			reduction
		};
	},
	[Primitive.Pool]: reshapeJit((st, { window, strides }) => pool(st, window, strides)),
	[Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
		let stX = poolTranspose(require_backend.ShapeTracker.fromShape(as.shape), inShape, window, strides);
		stX = stX.reshape([...inShape, require_backend.prod(stX.shape.slice(inShape.length))]);
		a = reshapeViews(a, (st) => st.compose(stX), true);
		const reduction = new require_backend.Reduction(a.dtype, require_backend.AluOp.Add, stX.shape[stX.shape.length - 1]);
		return {
			exp: [a],
			reduction
		};
	},
	[Primitive.Dot]([a, b], [as, bs]) {
		const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
		const [c] = k1.exp;
		const cs = promoteAvals(as, bs);
		return jitRules[Primitive.Reduce]([c], [cs], {
			op: require_backend.AluOp.Add,
			axis: [cs.ndim - 1]
		});
	},
	[Primitive.Conv]([a, b], [as, bs], params) {
		const [stX, stY] = prepareConv(require_backend.ShapeTracker.fromShape(as.shape), require_backend.ShapeTracker.fromShape(bs.shape), params);
		a = reshapeViews(a, (st) => st.compose(stX));
		b = reshapeViews(b, (st) => st.compose(stY));
		as = new ShapedArray(stX.shape, as.dtype, as.weakType);
		bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
		return jitRules[Primitive.Dot]([a, b], [as, bs], {});
	},
	[Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
	[Primitive.Where]: broadcastedJit(([cond, a, b]) => require_backend.AluExp.where(cond, a, b), { skipCastIdx: [0] }),
	[Primitive.Concatenate](exps, avals, { axis }) {
		const ndim$2 = avals[0].ndim;
		const sizes = avals.map((x) => x.shape[axis]);
		const finalSize = sizes.reduce((a, b) => a + b, 0);
		const { dtype: dtypeOut } = avals.map((x) => x.scalar()).reduce(promoteAvals);
		const makePadAxis = (start, end) => require_backend.range(ndim$2).map((i) => i === axis ? [start, end] : [0, 0]);
		let cum = 0;
		const src = [];
		for (let i = 0; i < exps.length; i++) {
			const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
			src.push(reshapeViews(require_backend.AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)));
			cum += sizes[i];
		}
		return { exp: [src.reduce(require_backend.AluExp.add)] };
	},
	[Primitive.Split]([a], [as], { axis, sizes }) {
		const exp$2 = [];
		let start = 0;
		for (const size$1 of sizes) {
			const slice = require_backend.range(as.ndim).map((d) => d === axis ? [start, start + size$1] : [0, as.shape[d]]);
			exp$2.push(reshapeViews(a, (st) => st.shrink(slice)));
			start += size$1;
		}
		return { exp: exp$2 };
	},
	[Primitive.RandomBits]: (keys, keyShapes, { shape: shape$1, mode }) => {
		const keyShape = keyShapes[0].shape;
		const mapping = (st) => {
			if (!require_backend.deepEqual(st.shape, shape$1)) return st.broadcast(shape$1, require_backend.range(st.shape.length, shape$1.length));
		};
		const k0 = reshapeViews(keys[0], mapping);
		const k1 = reshapeViews(keys[1], mapping);
		const c0 = require_backend.AluExp.u32(0);
		const c1 = require_backend.AluExp.mod(require_backend.AluExp.cast(require_backend.DType.Uint32, require_backend.AluVar.gidx), require_backend.AluExp.u32(Math.max(require_backend.prod(shape$1.slice(keyShape.length)), 1)));
		const exp$2 = require_backend.AluExp.threefry2x32(k0, k1, c0, c1, mode);
		return { exp: [exp$2] };
	},
	[Primitive.Gather]([x, ...indices], [xs, ...indicesShapes], { axis, outDim }) {
		const axisSet = new Set(axis);
		const indexShape = indicesShapes.map((c) => c.shape).reduce(require_backend.generalBroadcast);
		const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxAll = require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx);
		const idxNonaxis = [...idxAll];
		idxNonaxis.splice(outDim, indexShape.length);
		const src = [...idxNonaxis];
		for (let i = 0; i < xs.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, iexp] of indices.entries()) src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, reshapeViews(iexp, (st) => st.broadcast(finalShape, [...require_backend.range(outDim + indexShape.length - st.shape.length), ...require_backend.range(outDim + indexShape.length, finalShape.length)])));
		const [index, valid] = require_backend.ShapeTracker.fromShape(xs.shape).toAluExp(src);
		if (!valid.resolve()) throw new Error("internal: expected full validity mask in Gather");
		return { exp: [x.substitute({ gidx: index })] };
	},
	[Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
	[Primitive.Broadcast]: reshapeJit((st, { shape: shape$1, axis }) => st.broadcast(shape$1, axis)),
	[Primitive.Reshape]: reshapeJit((st, { shape: shape$1 }) => st.reshape(shape$1)),
	[Primitive.Flip]: reshapeJit((st, { axis }) => {
		const arg = require_backend.rep(st.shape.length, false);
		for (const ax of axis) arg[ax] = true;
		return st.flip(arg);
	}),
	[Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
	[Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
	[Primitive.Sort]: routineNoJit(),
	[Primitive.Argsort]: routineNoJit(),
	[Primitive.TriangularSolve]: routineNoJit(),
	[Primitive.Cholesky]: routineNoJit(),
	[Primitive.LU]: routineNoJit(),
	[Primitive.Jit]() {
		throw new Error("internal: Jit should have been flattened before JIT compilation");
	},
	[Primitive.Scan]() {
		throw new Error("internal: Scan is handled specially in jitCompile, not via jitRules");
	}
};
/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend, jaxpr) {
	const varToDefn = /* @__PURE__ */ new Map();
	const varToUsages = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		for (const v of eqn.outBinders) if (v instanceof Var) varToDefn.set(v, i);
		for (const input of eqn.inputs) if (input instanceof Var) {
			const usages = varToUsages.get(input);
			if (usages) usages.push(i);
			else varToUsages.set(input, [i]);
		}
	}
	const reducePrimitives = [
		Primitive.Reduce,
		Primitive.Dot,
		Primitive.Conv,
		Primitive.PoolTranspose
	];
	const reductionEpilogueEqns = /* @__PURE__ */ new Set();
	const reductionEndpointEqns = /* @__PURE__ */ new Set();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (reducePrimitives.includes(eqn.primitive)) {
			let head = i;
			while (true) {
				reductionEpilogueEqns.add(head);
				const outVar = jaxpr.eqns[head].outBinders[0];
				const usages = varToUsages.get(outVar) ?? [];
				if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;
				if (reductionEpilogueEqns.has(usages[0])) break;
				const nextEqn = jaxpr.eqns[usages[0]];
				switch (nextEqn.primitive) {
					case Primitive.Neg:
					case Primitive.Reciprocal:
					case Primitive.Floor:
					case Primitive.Ceil:
					case Primitive.StopGradient:
					case Primitive.Cast:
					case Primitive.Bitcast:
					case Primitive.Sin:
					case Primitive.Cos:
					case Primitive.Asin:
					case Primitive.Atan:
					case Primitive.Exp:
					case Primitive.Log:
					case Primitive.Erf:
					case Primitive.Erfc:
					case Primitive.Sqrt:
						head = usages[0];
						continue;
					case Primitive.Add:
					case Primitive.Mul:
					case Primitive.Idiv:
					case Primitive.Mod:
					case Primitive.Min:
					case Primitive.Max: {
						const otherInput = nextEqn.inputs.find((v) => v !== outVar);
						if (otherInput instanceof Lit || require_backend.deepEqual(require_backend.generalBroadcast(otherInput.aval.shape, outVar.aval.shape), outVar.aval.shape)) {
							head = usages[0];
							continue;
						}
						break;
					}
				}
				break;
			}
			reductionEndpointEqns.add(head);
		}
	}
	const blackNodes = /* @__PURE__ */ new Set();
	const p1NextBlack = /* @__PURE__ */ new Map();
	for (const v of jaxpr.outs) if (v instanceof Var) {
		blackNodes.add(v);
		p1NextBlack.set(v, v);
	}
	const heterogeneousViewPrimitives = [Primitive.RandomBits, Primitive.Gather];
	const needsCleanShapePrimitives = [Primitive.Concatenate, Primitive.Pad];
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		if (reductionEndpointEqns.has(i) || heterogeneousViewPrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive) || eqn.outBinders.some((v) => blackNodes.has(v))) {
			for (const v of eqn.outBinders) {
				blackNodes.add(v);
				p1NextBlack.set(v, v);
			}
			continue;
		}
		const reach = /* @__PURE__ */ new Set();
		let needsCleanOutput = false;
		outer: for (const v of eqn.outBinders) for (const j of varToUsages.get(v) ?? []) {
			if (needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) || routinePrimitives.has(jaxpr.eqns[j].primitive)) {
				needsCleanOutput = true;
				break outer;
			}
			for (const o of jaxpr.eqns[j].outBinders) {
				const u = p1NextBlack.get(o);
				if (u) reach.add(u);
			}
		}
		if (reach.size > 1 || needsCleanOutput) for (const v of eqn.outBinders) {
			blackNodes.add(v);
			p1NextBlack.set(v, v);
		}
		else if (reach.size === 1) {
			const b = reach.values().next().value;
			for (const v of eqn.outBinders) p1NextBlack.set(v, b);
		}
	}
	const p2Deps = /* @__PURE__ */ new Map();
	for (const v of jaxpr.inBinders) p2Deps.set(v, new Set([v]));
	let p2idx = 0;
	while (p2idx < jaxpr.eqns.length) {
		const eqn = jaxpr.eqns[p2idx++];
		const deps = [];
		for (const input of eqn.inputs) if (input instanceof Var) if (blackNodes.has(input)) deps.push(new Set([input]));
		else deps.push(p2Deps.get(input));
		else deps.push(/* @__PURE__ */ new Set());
		const depCounter = /* @__PURE__ */ new Map();
		for (const depSet of deps) for (const dep of depSet) depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
		if (depCounter.size > backend.maxArgs) {
			let maxUniqueDeps = 0;
			let assocInput = -1;
			for (let i = 0; i < eqn.inputs.length; i++) {
				const input = eqn.inputs[i];
				if (input instanceof Var && varToDefn.has(input)) {
					let uniqueDeps = 0;
					for (const dep of deps[i]) if (depCounter.get(dep) === 1) uniqueDeps++;
					if (uniqueDeps > maxUniqueDeps) {
						maxUniqueDeps = uniqueDeps;
						assocInput = i;
					}
				}
			}
			if (assocInput === -1) throw new Error(`internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`);
			const assocVar = eqn.inputs[assocInput];
			p2idx = varToDefn.get(assocVar);
			for (const out of jaxpr.eqns[p2idx++].outBinders) blackNodes.add(out);
		} else {
			const s = new Set(depCounter.keys());
			for (const out of eqn.outBinders) p2Deps.set(out, s);
		}
	}
	return blackNodes;
}
/**
* Try to prepare a native scan executable.
* Returns the executable if native scan is possible, null otherwise.
*
* Native scan is only supported when:
* 1. Backend is WASM or WebGPU
* 2. Body program contains exactly one execute step with a Kernel (no routines)
* 3. No constants (for MVP simplicity)
* 4. No reduction in the body kernel
*/
/**
* Try to prepare a batched scan for routine bodies (matmul, conv, etc.).
* Returns the PreparedBatchedScan params if possible, null otherwise.
*
* Batched scan is only supported when:
* 1. Backend is WebGPU
* 2. Body program contains exactly one execute step with a Routine (not Kernel)
* 3. MVP: No constants support
* 4. MVP: numCarry === numY (carry and output are the same)
*/
function tryPrepareBatchedScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, eqn, reverse) {
	if (backend.type !== "webgpu") {
		if (require_backend.DEBUG >= 2) console.log("Batched scan: skipped, unsupported backend");
		return null;
	}
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (executeSteps.length !== 1) {
		if (require_backend.DEBUG >= 2) console.log(`Batched scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`);
		return null;
	}
	const execStep = executeSteps[0];
	if (!(execStep.source instanceof require_backend.Routine)) {
		if (require_backend.DEBUG >= 2) console.log("Batched scan: skipped, not a Routine");
		return null;
	}
	const bodyRoutine = execStep.source;
	if (numCarry !== numY) {
		if (require_backend.DEBUG >= 2) console.log(`Batched scan: skipped, numCarry=${numCarry} !== numY=${numY}`);
		return null;
	}
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry).map((v) => v.aval);
	const carrySizes = carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const xsElemStrides = xAvals.map((a) => a.size);
	const ysElemStrides = carryAvals.map((a) => a.size);
	const webgpuBackend = backend;
	if (typeof webgpuBackend.prepareRoutineSync !== "function") {
		if (require_backend.DEBUG >= 2) console.log("Batched scan: skipped, backend has no prepareRoutineSync");
		return null;
	}
	let bodyRoutineExe;
	try {
		bodyRoutineExe = webgpuBackend.prepareRoutineSync(bodyRoutine);
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Batched scan: prepareRoutineSync failed:", e$1);
		return null;
	}
	if (typeof webgpuBackend.prepareBatchedScan !== "function") {
		if (require_backend.DEBUG >= 2) console.log("Batched scan: skipped, backend has no prepareBatchedScan");
		return null;
	}
	const batchedScanParams = {
		length,
		carrySizes,
		xsElemStrides,
		ysElemStrides,
		bodyRoutine: bodyRoutineExe,
		numCarry,
		numX,
		numY,
		numConsts,
		reverse
	};
	try {
		const prepared = webgpuBackend.prepareBatchedScan(batchedScanParams);
		if (prepared) {
			if (require_backend.DEBUG >= 1) console.log(`Batched scan: SUCCESS! Using WebGPU batched scan for ${bodyRoutine.name}`);
		}
		return prepared;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Batched scan preparation failed:", e$1);
		return null;
	}
}
function tryPrepareNativeScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, eqn, reverse) {
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (executeSteps.length !== 1) {
		if (require_backend.DEBUG >= 2) console.log(`Native scan: skipped, ${executeSteps.length} execute steps`);
		return null;
	}
	const execStep = executeSteps[0];
	if (!(execStep.source instanceof require_backend.Kernel)) {
		if (require_backend.DEBUG >= 2) console.log("Native scan: skipped, not a Kernel");
		return null;
	}
	const reindexMap = execStep.inputs;
	const reindexedExp = execStep.source.exp.reindexGids(reindexMap);
	const bodyKernel = new require_backend.Kernel(bodyJaxpr.inBinders.length, execStep.source.size, reindexedExp, execStep.source.reduction);
	if (numCarry !== numY) {
		if (require_backend.DEBUG >= 2) console.log(`Native scan: skipped, numCarry=${numCarry} !== numY=${numY}`);
		return null;
	}
	const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
	const yOutSlots = bodyProgram.outputs.slice(numCarry);
	for (let i = 0; i < numCarry; i++) if (carryOutSlots[i] !== yOutSlots[i]) {
		if (require_backend.DEBUG >= 2) console.log(`Native scan: skipped, carry[${i}] slot ${carryOutSlots[i]} !== Y[${i}] slot ${yOutSlots[i]}`);
		return null;
	}
	const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry, numConsts + numCarry + numX).map((v) => v.aval);
	const constSizes = constAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const carrySizes = carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const xsStrides = xAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const ysStrides = carrySizes.slice();
	const nativeBackend = backend;
	if (typeof nativeBackend.prepareNativeScan !== "function") {
		if (require_backend.DEBUG >= 2) console.log("Native scan: skipped, backend has no prepareNativeScan");
		return null;
	}
	const nativeScanParams = {
		length,
		numConsts,
		constSizes,
		carrySizes,
		xsStrides,
		ysStrides,
		bodyKernel,
		numCarry,
		reverse
	};
	try {
		const exe = nativeBackend.prepareNativeScan(nativeScanParams);
		if (exe) {
			if (require_backend.DEBUG >= 1) console.log(`Native scan: SUCCESS! Using ${backend.type.toUpperCase()} native scan loop`);
		}
		return exe;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Native scan preparation failed:", e$1);
		return null;
	}
}
/**
* Try to prepare a multi-kernel native scan for WASM.
* This handles scan bodies with multiple independent kernels (e.g., 2 matmuls).
*/
function tryPrepareNativeScanMulti(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse) {
	if (backend.type !== "wasm" && backend.type !== "webgpu") {
		if (require_backend.DEBUG >= 1) console.log(`[multi-scan] skipped, backend=${backend.type} (need wasm or webgpu)`);
		return null;
	}
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (require_backend.DEBUG >= 1) console.log(`[multi-scan] executeSteps.length=${executeSteps.length}, numCarry=${numCarry}`);
	if (!executeSteps.every((s) => s.source instanceof require_backend.Kernel)) {
		if (require_backend.DEBUG >= 1) console.log("[multi-scan] skipped, has non-Kernel execute steps");
		return null;
	}
	if (numCarry !== numY) {
		if (require_backend.DEBUG >= 1) console.log(`[multi-scan] skipped, numCarry=${numCarry} !== numY=${numY}`);
		return null;
	}
	const carryOutSlotsPre = bodyProgram.outputs.slice(0, numCarry);
	const yOutSlots = bodyProgram.outputs.slice(numCarry);
	for (let i = 0; i < numCarry; i++) if (carryOutSlotsPre[i] !== yOutSlots[i]) {
		if (require_backend.DEBUG >= 1) console.log(`[multi-scan] skipped, carry[${i}] slot ${carryOutSlotsPre[i]} !== Y[${i}] slot ${yOutSlots[i]}`);
		return null;
	}
	if (executeSteps.length !== numCarry) {
		if (require_backend.DEBUG >= 2) console.log(`Multi-kernel native scan: skipped, ${executeSteps.length} kernels != ${numCarry} carries`);
		return null;
	}
	const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
	const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry, numConsts + numCarry + numX).map((v) => v.aval);
	const constSizes = constAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const carrySizes = carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const xsStrides = xAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const ysStrides = carrySizes.slice();
	const steps = [];
	for (const execStep of executeSteps) {
		const kernel = execStep.source;
		const outSlot = execStep.outputs[0];
		const carryIdx = carryOutSlots.indexOf(outSlot);
		if (carryIdx === -1) {
			if (require_backend.DEBUG >= 2) console.log("Multi-kernel native scan: output slot not in carry outputs");
			return null;
		}
		const reindexMap = execStep.inputs;
		const reindexedExp = kernel.exp.reindexGids(reindexMap);
		const reindexedKernel = new require_backend.Kernel(bodyJaxpr.inBinders.length, kernel.size, reindexedExp, kernel.reduction);
		steps.push({
			kernel: reindexedKernel,
			inputs: execStep.inputs,
			outputCarryIdx: carryIdx,
			outputSize: carrySizes[carryIdx]
		});
	}
	const nativeBackend = backend;
	if (typeof nativeBackend.prepareNativeScanMulti !== "function") {
		if (require_backend.DEBUG >= 2) console.log("Multi-kernel native scan: backend has no prepareNativeScanMulti");
		return null;
	}
	const params = {
		length,
		numConsts,
		constSizes,
		numCarry,
		carrySizes,
		numX,
		xsStrides,
		numY,
		ysStrides,
		steps,
		reverse
	};
	try {
		const exe = nativeBackend.prepareNativeScanMulti(params);
		if (exe) {
			if (require_backend.DEBUG >= 1) console.log(`Multi-kernel native scan: SUCCESS! Using ${backend.type.toUpperCase()} native scan with ${steps.length} kernels`);
		}
		return exe;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Multi-kernel native scan preparation failed:", e$1);
		return null;
	}
}
/**
* Try to prepare a general native scan for bodies with data dependencies and numCarry !== numY.
* This handles complex scan bodies like the Kalman filter where there are multiple interdependent
* kernel steps and the number of carry values differs from the number of Y outputs.
* Also handles Routine steps (Cholesky, Sort) embedded in the scan loop.
*/
function tryPrepareNativeScanGeneral(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse) {
	if (backend.type !== "wasm") {
		if (require_backend.DEBUG >= 1) console.log(`[general-scan] skipped, backend=${backend.type} (need wasm)`);
		return null;
	}
	if (require_backend.DEBUG >= 2) console.log(`[general-scan] trying with backend=${backend.type}, numCarry=${numCarry}, numY=${numY}`);
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (executeSteps.length === 0) {
		if (require_backend.DEBUG >= 1) console.log("[general-scan] skipped, no execute steps");
		return null;
	}
	const usedRoutines = /* @__PURE__ */ new Set();
	for (const step of executeSteps) if (step.source instanceof require_backend.Routine) {
		const routineName = step.source.name;
		if (routineName !== require_backend.Routines.Cholesky && routineName !== require_backend.Routines.Sort) {
			if (require_backend.DEBUG >= 1) console.log(`[general-scan] skipped, unsupported routine: ${step.source.name}`);
			return null;
		}
		usedRoutines.add(routineName);
	}
	if (require_backend.DEBUG >= 1) {
		const routineNames = [...usedRoutines].map((r) => require_backend.Routines[r]);
		console.log(`[general-scan] Analyzing body: ${executeSteps.length} execute steps, numCarry=${numCarry}, numY=${numY}` + (routineNames.length > 0 ? `, routines: ${routineNames.join(", ")}` : ""));
	}
	const numInputs = numConsts + numCarry + numX;
	const numInternal = executeSteps.length;
	const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry, numConsts + numCarry + numX).map((v) => v.aval);
	const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
	const constSizes = constAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const carrySizes = carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const xsStrides = xAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const ysStrides = yAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const slotToInternal = /* @__PURE__ */ new Map();
	for (let i = 0; i < executeSteps.length; i++) {
		const step = executeSteps[i];
		for (const outSlot of step.outputs) slotToInternal.set(outSlot, i);
	}
	const internalSizes = [];
	for (let i = 0; i < executeSteps.length; i++) {
		const source = executeSteps[i].source;
		if (source instanceof require_backend.Kernel) internalSizes.push(source.size * require_backend.byteWidth(source.dtype));
		else {
			const routine = source;
			const outShape = routine.type.outputShapes[0];
			const outDtype = routine.type.outputDtypes[0];
			internalSizes.push(require_backend.prod(outShape) * require_backend.byteWidth(outDtype));
		}
	}
	let auxBufferSize = 0;
	let elementSize = 4;
	for (const step of executeSteps) if (step.source instanceof require_backend.Routine) {
		const routine = step.source;
		const dtype = routine.type.inputDtypes[0];
		elementSize = require_backend.byteWidth(dtype);
		if (routine.name === require_backend.Routines.Sort) {
			const inputShape = routine.type.inputShapes[0];
			const sortDim = inputShape[inputShape.length - 1];
			auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
		}
	}
	const steps = [];
	for (let i = 0; i < executeSteps.length; i++) {
		const step = executeSteps[i];
		const source = step.source;
		const inputSlots = [];
		for (const inputId of step.inputs) if (inputId < numInputs) inputSlots.push(inputId);
		else {
			const internalIdx = slotToInternal.get(inputId);
			if (internalIdx === void 0) {
				if (require_backend.DEBUG >= 1) console.log(`[general-scan] skipped, input ${inputId} not found in slot mapping`);
				return null;
			}
			inputSlots.push(numInputs + internalIdx);
		}
		if (source instanceof require_backend.Kernel) {
			const reindexMap = inputSlots;
			const reindexedExp = source.exp.reindexGids(reindexMap);
			const reindexedReduction = source.reduction?.reindexGids(reindexMap);
			const reindexedKernel = new require_backend.Kernel(numInputs + numInternal, source.size, reindexedExp, reindexedReduction);
			steps.push({
				source: reindexedKernel,
				inputSlots,
				outputInternalIdx: i
			});
		} else steps.push({
			source,
			inputSlots,
			outputInternalIdx: i
		});
	}
	const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
	const carryInputSlots = bodyProgram.inputs.slice(numConsts, numConsts + numCarry);
	const carryOutSources = [];
	for (const slot of carryOutSlots) {
		const carryIdx = carryInputSlots.indexOf(slot);
		if (carryIdx !== -1) {
			carryOutSources.push({
				type: "passthrough",
				carryIdx
			});
			continue;
		}
		const internalIdx = slotToInternal.get(slot);
		if (internalIdx === void 0) {
			if (require_backend.DEBUG >= 1) console.log(`[general-scan] skipped, carry output slot ${slot} not produced by any execute step`);
			return null;
		}
		carryOutSources.push({
			type: "internal",
			internalIdx
		});
	}
	const yOutputSlots = bodyProgram.outputs.slice(numCarry);
	const yOutputSources = [];
	for (const slot of yOutputSlots) {
		const carryIdx = carryInputSlots.indexOf(slot);
		if (carryIdx !== -1) {
			yOutputSources.push({
				type: "passthrough",
				carryIdx
			});
			continue;
		}
		const internalIdx = slotToInternal.get(slot);
		if (internalIdx === void 0) {
			if (require_backend.DEBUG >= 1) console.log(`[general-scan] skipped, Y output slot ${slot} not found`);
			return null;
		}
		yOutputSources.push({
			type: "internal",
			internalIdx
		});
	}
	const nativeBackend = backend;
	if (typeof nativeBackend.prepareNativeScanGeneral !== "function") {
		if (require_backend.DEBUG >= 2) console.log("[general-scan] backend has no prepareNativeScanGeneral");
		return null;
	}
	const params = {
		length,
		numConsts,
		constSizes,
		numCarry,
		carrySizes,
		numX,
		xsStrides,
		numY,
		ysStrides,
		internalSizes,
		steps,
		carryOutSources,
		yOutputSources,
		reverse,
		auxBufferSize,
		elementSize
	};
	try {
		const exe = nativeBackend.prepareNativeScanGeneral(params);
		if (exe) {
			if (require_backend.DEBUG >= 1) {
				const hasRoutines = steps.some((s) => s.source instanceof require_backend.Routine);
				console.log(`[general-scan] SUCCESS! Using ${backend.type.toUpperCase()} general native scan with ${steps.length} steps` + (hasRoutines ? " (includes routines)" : ""));
			}
			return {
				executable: exe,
				internalSizes,
				params
			};
		}
		return null;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("[general-scan] preparation failed:", e$1);
		return null;
	}
}

//#endregion
//#region src/frontend/array.ts
const JsArray$1 = globalThis.Array;
const inlineArrayLimit = 128;
/** Version of pureArray with fudged types. */
const fudgeArray = pureArray;
/**
* An executable operation that will be dispatched to the backend.
*
* This holds a reference to all input buffers used in the operation. After the
* operation is dispatched, the references should be released.
*/
var PendingExecute = class {
	prepared = null;
	submitted = false;
	#promise = null;
	#rc = 1;
	constructor(backend, source, inputs, outputs) {
		this.backend = backend;
		this.source = source;
		this.inputs = inputs;
		this.outputs = outputs;
		for (const slot of inputs) this.backend.incRef(slot);
		for (const slot of outputs) this.backend.incRef(slot);
	}
	updateRc(delta) {
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		this.#rc += delta;
		if (this.#rc <= 0 && !this.submitted) {
			for (const slot of this.inputs) this.backend.decRef(slot);
			for (const slot of this.outputs) this.backend.decRef(slot);
		}
	}
	async prepare() {
		if (this.prepared) return;
		if (this.#promise) {
			await this.#promise;
			return;
		}
		this.#promise = (async () => {
			if (this.source instanceof require_backend.Kernel) this.prepared = await this.backend.prepareKernel(this.source);
			else this.prepared = await this.backend.prepareRoutine(this.source);
		})();
		await this.#promise;
	}
	prepareSync() {
		if (this.prepared) return;
		if (this.source instanceof require_backend.Kernel) this.prepared = this.backend.prepareKernelSync(this.source);
		else this.prepared = this.backend.prepareRoutineSync(this.source);
	}
	submit() {
		if (this.submitted) return;
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		if (!this.prepared) throw new Error("internal: Not prepared yet");
		this.submitted = true;
		this.backend.dispatch(this.prepared, this.inputs, this.outputs);
		for (const slot of this.inputs) this.backend.decRef(slot);
		for (const slot of this.outputs) this.backend.decRef(slot);
	}
};
/**
* A multidimensional numeric array with data stored on CPU or GPU.
*
* This is the library's core data type. Equivalent to `jax.Array` from JAX, or
* `torch.Tensor`.
*
* Not to be confused with the JavaScript "Array" constructor. Avoid importing
* this into your code's namespace if you're already using the JavaScript
* "Array" type by name.
*/
var Array$1 = class Array$1 extends Tracer {
	#dtype;
	#weakType;
	#source;
	#st;
	#backend;
	#committed;
	#rc;
	#pendingSet;
	/**
	* @ignore
	* Constructs an array from source, shape and backend. Note that if the source
	* is a backend `Slot`, this constructor _takes ownership_ of the slot. It
	* will be freed when the array is disposed.
	*/
	constructor(args) {
		super(baseArrayTrace);
		this.#dtype = args.dtype;
		this.#weakType = args.weakType;
		this.#source = args.source;
		this.#st = args.st;
		this.#backend = args.backend;
		this.#committed = args.committed;
		this.#rc = 1;
		this.#pendingSet = new Set(args.pending);
		if (this.#pendingSet.size === 0) this.#pendingSet = null;
		else if (this.#source instanceof require_backend.AluExp) throw new Error("internal: AluExp source cannot have pending executes");
	}
	/** @ignore */
	get aval() {
		return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
	}
	/** Return a simple string representation of the array's dimensions. */
	toString() {
		return `Array:${this.#dtype}[${this.shape.join(",")}]`;
	}
	get device() {
		return this.#backend.type;
	}
	#check() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
	}
	/** Construct an array, copying fields from `this`. */
	#newArrayFrom(args) {
		return new Array$1({
			source: args.source ?? this.#source,
			st: args.st ?? this.#st,
			dtype: args.dtype ?? this.#dtype,
			weakType: this.#weakType,
			backend: args.backend ?? this.#backend,
			committed: args.committed ?? this.#committed,
			pending: args.pending ?? this.#pending ?? void 0
		});
	}
	get ref() {
		this.#check();
		this.#rc++;
		return this;
	}
	/** Get the current reference count (for debugging memory management). */
	get refCount() {
		return this.#rc;
	}
	dispose() {
		this.#check();
		if (--this.#rc === 0) {
			for (const exe of this.#pending) exe.updateRc(-1);
			if (typeof this.#source === "number") this.#backend.decRef(this.#source);
		}
	}
	/** Get the pending executes as a list, trimming if already submitted. */
	get #pending() {
		if (!this.#pendingSet) return [];
		for (const p of this.#pendingSet) if (p.submitted) this.#pendingSet.delete(p);
		if (this.#pendingSet.size === 0) {
			this.#pendingSet = null;
			return [];
		} else return [...this.#pendingSet];
	}
	/**
	* Convert this array into a primitive value.
	*
	* This only works for scalars (0-dimensional arrays). It lets you get values
	* "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
	* evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
	*
	* This method is also called for `==` equality.
	*/
	[Symbol.toPrimitive]() {
		if (this.ndim === 0) return this.dataSync()[0];
		else throw new Error(`Cannot convert non-scalar array to primitive: ${this.toString()}`);
	}
	#reshape(st) {
		this.#check();
		const pending = this.#pending;
		for (const exe of pending) exe.updateRc(1);
		if (typeof this.#source === "number") this.#backend.incRef(this.#source);
		const ar = this.#newArrayFrom({
			st,
			pending
		});
		this.dispose();
		return ar;
	}
	/**
	* Underlying implementation of the Gather primitive. This indexes an array
	* and extracts slices based on indices in other integer arrays.
	*/
	#gather(indices, axis, outDim) {
		this.#check();
		const axisSet = new Set(axis);
		if (axisSet.size !== axis.length) throw new TypeError("Gather axis must not have duplicates");
		if (indices.some((a) => a.#committed && a.#backend !== this.#backend)) throw new TypeError(`Gather indices must have the same backend: ${this.#backend.type}`);
		indices = indices.map((ar) => ar._putSync(this.#backend));
		indices = Array$1.#broadcastArrays(indices);
		const indexShape = indices[0].shape;
		const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxAll = require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx);
		const idxNonaxis = [...idxAll];
		const idxAxis = idxNonaxis.splice(outDim, indexShape.length);
		const inputs = [];
		const src = [...idxNonaxis];
		for (let i = 0; i < this.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, ar] of indices.entries()) if (ar.#source instanceof require_backend.AluExp) src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, require_backend.accessorAluExp(ar.#source, ar.#st, idxAxis));
		else {
			let gid = inputs.indexOf(ar.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(ar.#source);
			}
			src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis));
		}
		let exp$2;
		if (this.#source instanceof require_backend.AluExp) exp$2 = require_backend.accessorAluExp(this.#source, this.#st, src);
		else {
			let gid = inputs.indexOf(this.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(this.#source);
			}
			exp$2 = require_backend.accessorGlobal(this.#dtype, gid, this.#st, src);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(finalShape), exp$2);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		this.dispose();
		for (const ar of indices) ar.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(finalShape),
			pending
		});
	}
	/** Move axes to the rightmost dimension of the shape. */
	#moveAxesDown(axis) {
		this.#check();
		if (axis.length === 0) return this.reshape(this.shape.concat(1));
		const newShape = [];
		const keptAxes = [];
		const shiftedAxes = [];
		for (let i = 0; i < this.#st.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(this.#st.shape[i]);
		}
		newShape.push(-1);
		return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
	}
	#transpose(perm) {
		this.#check();
		if (!require_backend.isPermutation(perm, this.ndim)) throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
		return this.#reshape(this.#st.permute(perm));
	}
	#unary(op, dtypeOutput) {
		const weakType = !dtypeOutput && this.#weakType;
		dtypeOutput ??= this.#dtype;
		this.#check();
		if (this.#source instanceof require_backend.AluExp) {
			const exp$3 = new require_backend.AluExp(op, dtypeOutput, [this.#source]);
			this.dispose();
			return this.#newArrayFrom({
				source: exp$3.simplify(),
				dtype: dtypeOutput,
				weakType
			});
		}
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		const exp$2 = new require_backend.AluExp(op, dtypeOutput, [require_backend.AluExp.globalView(this.#dtype, 0, this.#st, indices)]);
		const kernel = new require_backend.Kernel(1, this.#st.size, exp$2);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, [this.#source], [output]));
		this.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(this.shape),
			dtype: dtypeOutput,
			weakType,
			pending
		});
	}
	#binary(op, other) {
		const custom = (src) => new require_backend.AluExp(op, src[0].dtype, src);
		return Array$1.#naryCustom(op, custom, [this, other]);
	}
	static #naryCustom(name, custom, arrays, { dtypeOverride, strongTypeOutput, reduceAxis } = {}) {
		const n = arrays.length;
		if (n === 0) throw new TypeError(`No inputs for ${name}`);
		for (const ar of arrays) ar.#check();
		let castDtype;
		let castWeakType = true;
		for (let i = 0; i < n; i++) if (dtypeOverride?.[i]) {
			if (arrays[i].#dtype !== dtypeOverride[i]) throw new TypeError(`Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`);
		} else if (castDtype === void 0) {
			castDtype = arrays[i].#dtype;
			castWeakType = arrays[i].#weakType;
		} else ({dtype: castDtype, weakType: castWeakType} = promoteAvals(new ShapedArray([], castDtype, castWeakType), arrays[i].aval.scalar()));
		const weakType = castWeakType && !strongTypeOutput;
		const { backend, committed } = Array$1.#computeBackend(name, arrays);
		arrays = arrays.map((ar) => ar._putSync(backend));
		arrays = Array$1.#broadcastArrays(arrays);
		const newShape = [...arrays[0].shape];
		if (arrays.every((ar) => ar.#source instanceof require_backend.AluExp) && !reduceAxis) {
			const sources = arrays.map((ar, i) => {
				if (!dtypeOverride?.[i]) return require_backend.AluExp.cast(castDtype, ar.#source);
				else return ar.#source;
			});
			if (arrays.every((ar) => require_backend.deepEqual(ar.#st, arrays[0].#st))) {
				const exp$4 = custom(sources);
				arrays.forEach((ar) => ar.dispose());
				return new Array$1({
					source: exp$4.simplify(),
					st: arrays[0].#st,
					dtype: exp$4.dtype,
					weakType,
					backend,
					committed
				});
			}
			const exp$3 = custom(arrays.map((ar, i) => {
				const src$1 = sources[i];
				if (ar.#st.contiguous) return src$1;
				return require_backend.accessorAluExp(src$1, ar.#st, require_backend.unravelAlu(newShape, require_backend.AluVar.idx));
			}));
			const st = require_backend.ShapeTracker.fromShape(newShape);
			arrays.forEach((ar) => ar.dispose());
			return new Array$1({
				source: exp$3.simplify(),
				st,
				dtype: exp$3.dtype,
				weakType,
				backend,
				committed
			});
		}
		let indices;
		if (!reduceAxis) indices = require_backend.unravelAlu(newShape, require_backend.AluVar.gidx);
		else {
			const contractedShape = newShape.slice(0, -1);
			indices = [...require_backend.unravelAlu(contractedShape, require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		}
		const inputs = [];
		const src = [];
		for (const [i, ar] of arrays.entries()) {
			let nextSrc;
			if (ar.#source instanceof require_backend.AluExp) nextSrc = require_backend.accessorAluExp(ar.#source, ar.#st, indices);
			else {
				let gid = inputs.indexOf(ar.#source);
				if (gid === -1) {
					gid = inputs.length;
					inputs.push(ar.#source);
				}
				nextSrc = require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
			}
			if (!dtypeOverride?.[i]) nextSrc = require_backend.AluExp.cast(castDtype, nextSrc);
			src.push(nextSrc);
		}
		const exp$2 = custom(src);
		let re = void 0;
		if (reduceAxis) {
			const [axisSize] = newShape.splice(-1, 1);
			re = new require_backend.Reduction(exp$2.dtype, require_backend.AluOp.Add, axisSize);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(newShape), exp$2, re);
		const output = backend.malloc(kernel.bytes);
		const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
		for (const exe of pending) exe.updateRc(1);
		pending.add(new PendingExecute(backend, kernel, inputs, [output]));
		arrays.forEach((ar) => ar.dispose());
		return new Array$1({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			dtype: kernel.dtype,
			weakType,
			backend,
			committed,
			pending
		});
	}
	/** Reduce the last dimension of the array by an operation. */
	#reduce(op) {
		const shape$1 = this.shape;
		const reduction = new require_backend.Reduction(this.#dtype, op, shape$1[shape$1.length - 1]);
		const newShape = shape$1.slice(0, -1);
		const newSize = require_backend.prod(newShape);
		const indices = [...require_backend.unravelAlu(newShape, require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		let exp$2;
		const inputs = [];
		if (this.#source instanceof require_backend.AluExp) exp$2 = require_backend.accessorAluExp(this.#source, this.#st, indices);
		else {
			inputs.push(this.#source);
			exp$2 = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
		}
		const kernel = new require_backend.Kernel(inputs.length, newSize, exp$2, reduction);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		this.dispose();
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			pending
		});
	}
	/** Apply an operation with custom lowering to this array. */
	static #routine(prim) {
		return (arrays, params) => {
			const { backend, committed } = Array$1.#computeBackend(prim, arrays);
			for (const ar of arrays) ar.#realize();
			const avals = arrays.map((ar) => ar.aval);
			const avalsOut = abstractEvalRules[prim](avals, params);
			const routine = new require_backend.Routine(routinePrimitives.get(prim), {
				inputShapes: avals.map((a) => a.shape),
				inputDtypes: avals.map((a) => a.dtype),
				outputShapes: avalsOut.map((a) => a.shape),
				outputDtypes: avalsOut.map((a) => a.dtype)
			}, params);
			const inputs = arrays.map((ar) => ar.#source);
			const outputs = avalsOut.map((x) => backend.malloc(require_backend.byteWidth(x.dtype) * x.size));
			const pending = arrays.flatMap((ar) => ar.#pending);
			for (const exe of pending) exe.updateRc(+outputs.length);
			pending.push(new PendingExecute(backend, routine, inputs, outputs));
			pending[pending.length - 1].updateRc(+outputs.length - 1);
			arrays.forEach((ar) => ar.dispose());
			return outputs.map((output, i) => new Array$1({
				source: output,
				st: require_backend.ShapeTracker.fromShape(avalsOut[i].shape),
				dtype: avalsOut[i].dtype,
				weakType: avalsOut[i].weakType,
				backend,
				committed,
				pending
			}));
		};
	}
	/**
	* Normalizes this array into one backed by a `Slot`.
	*
	* This mutates the array in-place, turning it into an equivalent array whose
	* source is actual, contiguous data on device.
	*
	* Calling this twice is a no-op.
	*/
	#realize() {
		this.#check();
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		if (this.#source instanceof require_backend.AluExp) {
			const exp$2 = require_backend.accessorAluExp(this.#source, this.#st, indices);
			const kernel = new require_backend.Kernel(0, this.#st.size, exp$2);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [], [output]);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet = new Set([pendingItem]);
		} else {
			if (this.#st.contiguous) return;
			const exp$2 = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
			const kernel = new require_backend.Kernel(1, this.#st.size, exp$2);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [this.#source], [output]);
			this.#backend.decRef(this.#source);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet ??= /* @__PURE__ */ new Set();
			this.#pendingSet.add(pendingItem);
		}
	}
	#dataInline() {
		this.#check();
		if (!(this.#source instanceof require_backend.AluExp)) throw new Error("internal: #dataInline called on non-AluExp source");
		const ar = this.#newArrayFrom({ backend: require_backend.getBackend("cpu") });
		this.dispose();
		return ar.dataSync();
	}
	static #broadcastArrays(arrays) {
		if (arrays.length === 0) throw new Error("Need at least one array to broadcast");
		if (arrays.length === 1) return arrays;
		const newShape = arrays.map((a) => a.shape).reduce(require_backend.generalBroadcast);
		return arrays.map((ar) => {
			if (require_backend.deepEqual(ar.shape, newShape)) return ar;
			return ar.#reshape(ar.#st.broadcast(newShape, require_backend.range(newShape.length - ar.ndim)));
		});
	}
	static #computeBackend(name, arrays) {
		const committed = arrays.filter((ar) => ar.#committed);
		if (committed.length > 0) {
			const backend = committed[0].#backend;
			for (const ar of committed) if (ar.#backend !== backend) throw new Error(`Device mismatch in ${name} between committed arrays on (${backend.type}, ${ar.#backend.type}), please move to the same device with devicePut()`);
			return {
				backend,
				committed: true
			};
		} else {
			const backend = arrays.length > 0 ? arrays[0].#backend : require_backend.getBackend();
			return {
				backend,
				committed: false
			};
		}
	}
	/** Realize the array and return it as data. */
	async data() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = await this.#backend.read(this.#source, 0, byteCount);
		this.dispose();
		return require_backend.dtypedArray(this.dtype, buf);
	}
	/**
	* Wait for this array to finish evaluation.
	*
	* Operations and data loading in jax-js are lazy, so this function ensures
	* that pending operations are dispatched and fully executed before it
	* returns.
	*
	* If you are mapping from `data()` or `dataSync()`, it will also trigger
	* dispatch of operations as well.
	*
	* **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
	* asynchronously for multiple arrays.
	*/
	async blockUntilReady() {
		this.#check();
		if (this.#source instanceof require_backend.AluExp) return this;
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		await this.#backend.read(this.#source, 0, 0);
		return this;
	}
	/**
	* Realize the array and return it as data. This is a sync variant and not
	* recommended for performance reasons, as it will block rendering.
	*/
	dataSync() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		for (const p of this.#pending) {
			p.prepareSync();
			p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = this.#backend.readSync(this.#source, 0, byteCount);
		this.dispose();
		return require_backend.dtypedArray(this.dtype, buf);
	}
	/**
	* Convert this array into a JavaScript object.
	*
	* This is a blocking operation that will compile all of the shaders and wait
	* for execution to complete, synchronously. No other JavaScript code on the
	* site will be run during shader execution.
	*
	* To avoid blocking, prefer `jsAsync()` when possible.
	*/
	js() {
		return dataToJs(this.dtype, this.dataSync(), this.shape);
	}
	/** Convert this array into a JavaScript object, asynchronously. */
	async jsAsync() {
		return dataToJs(this.dtype, await this.data(), this.shape);
	}
	/**
	* Copy an element of an array to a numeric scalar and return it.
	*
	* Throws an error if the array does not have a single element. The array must
	* either be rank-0, or all dimensions of the shape are 1.
	*/
	item() {
		if (this.size !== 1) throw new Error(`item() can only be called on arrays of size 1`);
		return this.dataSync()[0];
	}
	/** @private Internal plumbing method for Array / Tracer ops. */
	static _implRules() {
		return {
			[Primitive.Add]([x, y]) {
				return [x.#binary(require_backend.AluOp.Add, y)];
			},
			[Primitive.Mul]([x, y]) {
				return [x.#binary(require_backend.AluOp.Mul, y)];
			},
			[Primitive.Idiv]([x, y]) {
				return [x.#binary(require_backend.AluOp.Idiv, y)];
			},
			[Primitive.Mod]([x, y]) {
				return [x.#binary(require_backend.AluOp.Mod, y)];
			},
			[Primitive.Min]([x, y]) {
				return [x.#binary(require_backend.AluOp.Min, y)];
			},
			[Primitive.Max]([x, y]) {
				return [x.#binary(require_backend.AluOp.Max, y)];
			},
			[Primitive.Neg]([x]) {
				return [zerosLike$1(x.ref).#binary(require_backend.AluOp.Sub, x)];
			},
			[Primitive.Reciprocal]([x]) {
				return [x.#unary(require_backend.AluOp.Reciprocal)];
			},
			[Primitive.Floor]([x]) {
				return [x.#unary(require_backend.AluOp.Floor)];
			},
			[Primitive.Ceil]([x]) {
				return [x.#unary(require_backend.AluOp.Ceil)];
			},
			[Primitive.StopGradient]([x]) {
				return [x];
			},
			[Primitive.Cast]([x], { dtype }) {
				return [x.#unary(require_backend.AluOp.Cast, dtype)];
			},
			[Primitive.Bitcast]([x], { dtype }) {
				if (x.dtype === require_backend.DType.Bool || dtype === require_backend.DType.Bool) throw new TypeError("Bitcast to/from bool is not allowed");
				if (x.dtype === dtype) return [x];
				if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
				if (x.#source instanceof require_backend.AluExp) return [x.#unary(require_backend.AluOp.Bitcast, dtype)];
				else {
					x.#backend.incRef(x.#source);
					const pending = x.#pending;
					for (const exe of pending) exe.updateRc(1);
					const y = x.#newArrayFrom({
						dtype,
						weakType: false,
						pending
					});
					x.dispose();
					return [y];
				}
			},
			[Primitive.Sin]([x]) {
				return [x.#unary(require_backend.AluOp.Sin)];
			},
			[Primitive.Cos]([x]) {
				return [x.#unary(require_backend.AluOp.Cos)];
			},
			[Primitive.Asin]([x]) {
				return [x.#unary(require_backend.AluOp.Asin)];
			},
			[Primitive.Atan]([x]) {
				return [x.#unary(require_backend.AluOp.Atan)];
			},
			[Primitive.Exp]([x]) {
				return [x.#unary(require_backend.AluOp.Exp)];
			},
			[Primitive.Log]([x]) {
				return [x.#unary(require_backend.AluOp.Log)];
			},
			[Primitive.Erf]([x]) {
				return [x.#unary(require_backend.AluOp.Erf)];
			},
			[Primitive.Erfc]([x]) {
				return [x.#unary(require_backend.AluOp.Erfc)];
			},
			[Primitive.Sqrt]([x]) {
				return [x.#unary(require_backend.AluOp.Sqrt)];
			},
			[Primitive.Reduce]([x], { op, axis }) {
				if (axis.length === 0) return [x];
				return [x.#moveAxesDown(axis).#reduce(op)];
			},
			[Primitive.Pool]([x], { window, strides }) {
				const st = pool(x.#st, window, strides);
				return [x.#reshape(st)];
			},
			[Primitive.PoolTranspose]([x], { inShape, window, strides }) {
				const n = inShape.length;
				let st = poolTranspose(x.#st, inShape, window, strides);
				st = st.reshape([...st.shape.slice(0, n), require_backend.prod(st.shape.slice(n))]);
				return [x.#reshape(st).#reduce(require_backend.AluOp.Add)];
			},
			[Primitive.Dot]([x, y]) {
				return [Array$1.#naryCustom("dot", ([x$1, y$1]) => require_backend.AluExp.mul(x$1, y$1), [x, y], { reduceAxis: true })];
			},
			[Primitive.Conv]([x, y], params) {
				checkConvShape(x.shape, y.shape, params);
				const [stX, stY] = prepareConv(x.#st, y.#st, params);
				return [Array$1.#naryCustom("conv", ([x$1, y$1]) => require_backend.AluExp.mul(x$1, y$1), [x.#reshape(stX), y.#reshape(stY)], { reduceAxis: true })];
			},
			[Primitive.Compare]([x, y], { op }) {
				const custom = ([x$1, y$1]) => aluCompare(x$1, y$1, op);
				return [Array$1.#naryCustom("compare", custom, [x, y], { strongTypeOutput: true })];
			},
			[Primitive.Where]([cond, x, y]) {
				const custom = ([cond$1, x$1, y$1]) => require_backend.AluExp.where(cond$1, x$1, y$1);
				return [Array$1.#naryCustom("where", custom, [
					cond,
					x,
					y
				], { dtypeOverride: [require_backend.DType.Bool] })];
			},
			[Primitive.Concatenate](xs, { axis }) {
				const ndim$2 = xs[0].ndim;
				const sizes = xs.map((x) => x.shape[axis]);
				const finalSize = sizes.reduce((a, b) => a + b, 0);
				const makePadAxis = (start, end) => require_backend.range(ndim$2).map((i) => i === axis ? [start, end] : [0, 0]);
				let cum = 0;
				const xsPadded = [];
				for (let i = 0; i < xs.length; i++) {
					const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
					xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
					cum += sizes[i];
				}
				const custom = (exps) => exps.reduce(require_backend.AluExp.add);
				return [Array$1.#naryCustom("concatenate", custom, xsPadded)];
			},
			[Primitive.Split]([x], { axis, sizes }) {
				const outputs = [];
				for (let i = 0, start = 0; i < sizes.length; i++) {
					const slice = require_backend.range(x.ndim).map((d) => d === axis ? [start, start + sizes[i]] : [0, x.shape[d]]);
					outputs.push(x.ref.#reshape(x.#st.shrink(slice)));
					start += sizes[i];
				}
				x.dispose();
				return outputs;
			},
			[Primitive.RandomBits]([k0, k1], { shape: shape$1, mode }) {
				const keyShape = k0.shape;
				const genShape = shape$1.slice(keyShape.length);
				const c0 = zeros(genShape, {
					dtype: require_backend.DType.Uint32,
					device: k0.device
				});
				const c1 = arange(0, require_backend.prod(genShape), 1, {
					dtype: require_backend.DType.Uint32,
					device: k0.device
				}).reshape(genShape);
				k0 = k0.#reshape(k0.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
				k1 = k1.#reshape(k1.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
				const custom = ([k0$1, k1$1, c0$1, c1$1]) => require_backend.AluExp.threefry2x32(k0$1, k1$1, c0$1, c1$1, mode);
				return [Array$1.#naryCustom("random_bits", custom, [
					k0,
					k1,
					c0,
					c1
				])];
			},
			[Primitive.Gather]([x, ...indices], { axis, outDim }) {
				return [x.#gather(indices, axis, outDim)];
			},
			[Primitive.Transpose]([x], { perm }) {
				return [x.#transpose(perm)];
			},
			[Primitive.Broadcast]([x], { shape: shape$1, axis }) {
				return [x.#reshape(x.#st.broadcast(shape$1, axis))];
			},
			[Primitive.Reshape]([x], { shape: shape$1 }) {
				return [x.#reshape(x.#st.reshape(shape$1))];
			},
			[Primitive.Flip]([x], { axis }) {
				const arg = require_backend.rep(x.ndim, false);
				for (const ax of axis) arg[ax] = true;
				return [x.#reshape(x.#st.flip(arg))];
			},
			[Primitive.Shrink]([x], { slice }) {
				return [x.#reshape(x.#st.shrink(slice))];
			},
			[Primitive.Pad]([x], { width }) {
				return [x.#reshape(x.#st.pad(width))];
			},
			[Primitive.Sort]: Array$1.#routine(Primitive.Sort),
			[Primitive.Argsort]: Array$1.#routine(Primitive.Argsort),
			[Primitive.TriangularSolve]: Array$1.#routine(Primitive.TriangularSolve),
			[Primitive.Cholesky]: Array$1.#routine(Primitive.Cholesky),
			[Primitive.LU]: Array$1.#routine(Primitive.LU),
			[Primitive.Jit](args, { jaxpr }) {
				if (jaxpr.inBinders.length !== args.length) throw new Error(`jit expects ${jaxpr.inBinders.length} args, got ${args.length}`);
				const { backend, committed } = Array$1.#computeBackend("jit", args);
				args = args.map((ar) => ar._putSync(backend));
				const jp = jitCompile(backend, jaxpr);
				const scanRunner = (bodyProgram, _backend, bodyJaxpr, length, numCarry, _numConsts, _numX, numY, reverse, constSlots, initCarrySlots, xsSlots, xsAvals, _outputSlots) => {
					const carryAvals = bodyJaxpr.inBinders.slice(constSlots.length, constSlots.length + numCarry).map((v) => v.aval);
					const constSlotsRealized = constSlots;
					const xs = xsSlots.map((slot, i) => new Array$1({
						source: slot,
						st: require_backend.ShapeTracker.fromShape(xsAvals[i].shape),
						dtype: xsAvals[i].dtype,
						weakType: xsAvals[i].weakType,
						backend,
						committed,
						pending: []
					}));
					const xSliceSts = xs.map((x) => {
						const sliceSts = [];
						const squeezedShape = x.shape.slice(1);
						for (let iter = 0; iter < length; iter++) {
							const i = reverse ? length - 1 - iter : iter;
							const slicePairs = x.shape.map((s, axis) => axis === 0 ? [i, i + 1] : [0, s]);
							sliceSts.push(x.#st.shrink(slicePairs).reshape(squeezedShape));
						}
						return sliceSts;
					});
					let carry = initCarrySlots.map((slot, i) => new Array$1({
						source: slot,
						st: require_backend.ShapeTracker.fromShape(carryAvals[i].shape),
						dtype: carryAvals[i].dtype,
						weakType: carryAvals[i].weakType,
						backend,
						committed,
						pending: []
					}));
					const ySlices = [];
					for (let j = 0; j < numY; j++) ySlices.push([]);
					const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);
					for (let i = 0; i < length; i++) {
						const xSlice = xs.map((x, xIdx) => x.ref.#reshape(xSliceSts[xIdx][i]));
						const carrySlots = carry.map((c) => c._realizeSource());
						const xSliceSlots = xSlice.map((x) => x._realizeSource());
						for (const x of xSlice) for (const exe of x.#pending) {
							exe.prepareSync();
							exe.submit();
						}
						if (require_backend.DEBUG >= 2) console.log(`[scanRunner] iter ${i}: carrySlots=${carrySlots}, xSliceSlots=${xSliceSlots}`);
						const { outputs: bodyOuts, pending: pending$1 } = bodyProgram.execute([
							...constSlotsRealized,
							...carrySlots,
							...xSliceSlots
						]);
						if (require_backend.DEBUG >= 2) console.log(`[scanRunner] iter ${i}: bodyOuts=${bodyOuts}, pending.length=${pending$1.length}`);
						for (const exe of pending$1) {
							exe.prepareSync();
							exe.submit();
						}
						xSlice.forEach((x) => x.dispose());
						const seenSlots = /* @__PURE__ */ new Set();
						const outArrays = bodyOuts.map((slot, j) => {
							if (seenSlots.has(slot)) backend.incRef(slot);
							else seenSlots.add(slot);
							return new Array$1({
								source: slot,
								st: require_backend.ShapeTracker.fromShape(bodyOutAvals[j].shape),
								dtype: bodyOutAvals[j].dtype,
								weakType: bodyOutAvals[j].weakType,
								backend,
								committed,
								pending: []
							});
						});
						const newCarry = outArrays.slice(0, numCarry);
						const ySlice = outArrays.slice(numCarry);
						for (let j = 0; j < numY; j++) ySlices[j].push(ySlice[j]);
						if (i > 0) {
							const oldCarrySlots = new Set(carry.map((c) => c._realizeSource()));
							for (const y of ySlice) {
								const slot = y._realizeSource();
								if (oldCarrySlots.has(slot)) backend.incRef(slot);
							}
							carry.forEach((c) => c.dispose());
						}
						carry = newCarry;
					}
					const stackedYs = ySlices.map((slices) => {
						const reshaped = slices.map((s) => {
							const expanded = s.ref.#reshape(s.#st.reshape([1, ...s.shape]));
							s.dispose();
							return expanded;
						});
						let stacked = reshaped[0];
						for (let i = 1; i < reshaped.length; i += 6) {
							const chunk = reshaped.slice(i, i + 6);
							stacked = concatenate$1([stacked, ...chunk], 0);
						}
						if (reverse) {
							const flipArg = require_backend.rep(stacked.ndim, false);
							flipArg[0] = true;
							const flipped = stacked.ref.#reshape(stacked.#st.flip(flipArg));
							stacked.dispose();
							return flipped;
						}
						return stacked;
					});
					const carryOutSlots = carry.map((c) => c._realizeSource());
					const yOutSlots = stackedYs.map((y) => y._realizeSource());
					const carryPending = carry.flatMap((c) => c.#pending);
					const ysPending = stackedYs.flatMap((y) => y.#pending);
					const finalPending = [...carryPending, ...ysPending];
					return {
						outputs: [...carryOutSlots, ...yOutSlots],
						pending: finalPending
					};
				};
				const realizedInputs = args.map((x) => x._realizeSource());
				const prevPending = [...new Set(args.flatMap((x) => x.#pending))];
				for (const exe of prevPending) {
					exe.prepareSync();
					exe.submit();
				}
				const { outputs, pending } = jp.execute(realizedInputs, scanRunner);
				for (const exe of pending) exe.updateRc(+outputs.length - 1);
				args.forEach((x) => x.dispose());
				return outputs.map((source, i) => {
					return new Array$1({
						source,
						st: require_backend.ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
						dtype: jaxpr.outs[i].aval.dtype,
						weakType: jaxpr.outs[i].aval.weakType,
						backend,
						committed,
						pending
					});
				});
			},
			[Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse }) {
				const consts = args.slice(0, numConsts);
				const initCarry = args.slice(numConsts, numConsts + numCarry);
				const xs = args.slice(numConsts + numCarry);
				const numX = xs.length;
				const numY = jaxpr.outs.length - numCarry;
				if (jaxpr.inBinders.length !== numConsts + numCarry + numX) throw new Error(`scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`);
				const xSliceSts = xs.map((x) => {
					const sliceSts = [];
					const squeezedShape = x.shape.slice(1);
					for (let iter = 0; iter < length; iter++) {
						const i = reverse ? length - 1 - iter : iter;
						const slicePairs = x.shape.map((s, axis) => axis === 0 ? [i, i + 1] : [0, s]);
						sliceSts.push(x.#st.shrink(slicePairs).reshape(squeezedShape));
					}
					return sliceSts;
				});
				const constViews = consts.map((c) => c.ref.#reshape(c.#st));
				const ySlices = [];
				for (let j = 0; j < numY; j++) ySlices.push([]);
				let carry = initCarry;
				for (let i = 0; i < length; i++) {
					const xSlice = xs.map((x, xIdx) => x.ref.#reshape(xSliceSts[xIdx][i]));
					const jaxprInputs = [
						...constViews.map((c) => c.ref),
						...carry.map((c) => c.ref),
						...xSlice
					];
					const outs = evalJaxpr(jaxpr, jaxprInputs);
					const newCarry = outs.slice(0, numCarry);
					const ySlice = outs.slice(numCarry);
					for (let j = 0; j < numY; j++) ySlices[j].push(ySlice[j]);
					if (i > 0) carry.forEach((c) => c.dispose());
					carry = newCarry;
				}
				initCarry.forEach((c) => c.dispose());
				xs.forEach((x) => x.dispose());
				consts.forEach((c) => c.dispose());
				constViews.forEach((c) => c.dispose());
				const stackedYs = ySlices.map((slices) => {
					const reshaped = slices.map((s) => {
						const expanded = s.ref.#reshape(s.#st.reshape([1, ...s.shape]));
						s.dispose();
						return expanded;
					});
					let stacked = reshaped[0];
					for (let i = 1; i < reshaped.length; i += 6) {
						const chunk = reshaped.slice(i, i + 6);
						stacked = concatenate$1([stacked, ...chunk], 0);
					}
					if (reverse) {
						const flipArg = require_backend.rep(stacked.ndim, false);
						flipArg[0] = true;
						const flipped = stacked.ref.#reshape(stacked.#st.flip(flipArg));
						stacked.dispose();
						return flipped;
					}
					return stacked;
				});
				return [...carry, ...stackedYs];
			}
		};
	}
	/** @private */
	_realizeSource() {
		this.#realize();
		return this.#source;
	}
	/** @private Put this array on a new backend, asynchronously. */
	async _put(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) {
			const ar = this.#newArrayFrom({
				backend,
				committed: true
			});
			this.dispose();
			return ar;
		} else {
			const data = await this.data();
			return arrayFromData(data, this.shape, {
				dtype: this.#dtype,
				device: backend.type
			}, this.#weakType);
		}
	}
	/** @private Put this array on a new backend, synchronously. */
	_putSync(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) {
			const ar = this.#newArrayFrom({
				backend,
				committed: true
			});
			this.dispose();
			return ar;
		} else {
			const data = this.dataSync();
			return arrayFromData(data, this.shape, {
				dtype: this.#dtype,
				device: backend.type
			}, this.#weakType);
		}
	}
};
/** Constructor for creating a new array from data. */
function array(values, { shape: shape$1, dtype, device } = {}) {
	if (values instanceof Tracer) {
		if (shape$1 && !require_backend.deepEqual(values.shape, shape$1)) values = values.reshape(shape$1);
		if (dtype && values.dtype !== dtype) values = values.astype(dtype);
		return values;
	} else if (ArrayBuffer.isView(values)) return arrayFromData(values, shape$1 ?? [values.length], {
		dtype,
		device
	});
	else {
		if (!shape$1) {
			shape$1 = [];
			let cur = values;
			while (JsArray$1.isArray(cur)) {
				shape$1.push(cur.length);
				cur = cur[0];
			}
		}
		const size$1 = require_backend.prod(shape$1);
		const flat = require_backend.recursiveFlatten(values);
		if (flat.length !== size$1) throw new Error(`Jagged shape: ${JSON.stringify(shape$1)} vs ${flat.length}`);
		if (size$1 === 0) return zeros(shape$1, {
			dtype,
			device
		});
		if (size$1 === 1) return full(shape$1, flat[0], {
			dtype,
			device
		});
		if (typeof flat[0] === "boolean") {
			dtype = dtype ?? require_backend.DType.Bool;
			const data = new Int32Array(flat.map((x) => x ? 1 : 0));
			return arrayFromData(data, shape$1, {
				dtype,
				device
			});
		} else {
			const weakType = dtype == void 0 && shape$1.length === 0;
			dtype = dtype ?? require_backend.DType.Float32;
			const data = require_backend.dtypedJsArray(dtype, flat);
			return arrayFromData(data, shape$1, {
				dtype,
				device
			}, weakType);
		}
	}
}
function arrayFromData(data, shape$1, { dtype, device }, weakType = false) {
	if (data instanceof Float32Array) {
		if (dtype && dtype !== require_backend.DType.Float32) throw new Error("Float32Array must have float32 type");
		dtype ??= require_backend.DType.Float32;
	} else if (data instanceof Int32Array) {
		if (dtype && dtype !== require_backend.DType.Int32 && dtype !== require_backend.DType.Bool) throw new Error("Int32Array must have int32 or bool type");
		dtype ??= require_backend.DType.Int32;
	} else if (data instanceof Uint32Array) {
		if (dtype && dtype !== require_backend.DType.Uint32) throw new Error("Uint32Array must have uint32 type");
		dtype ??= require_backend.DType.Uint32;
	} else if (data instanceof Float16Array) {
		if (dtype && dtype !== require_backend.DType.Float16) throw new Error("Float16Array must have float16 type");
		dtype ??= require_backend.DType.Float16;
	} else if (data instanceof Float64Array) {
		if (dtype && dtype !== require_backend.DType.Float64) throw new Error("Float64Array must have float64 type");
		dtype ??= require_backend.DType.Float64;
	} else throw new Error("Unsupported data array type: " + data.constructor.name);
	if (data.length < inlineArrayLimit) {
		let allEqual = true;
		for (let i = 1; i < data.length; i++) if (data[i] !== data[0]) {
			allEqual = false;
			break;
		}
		if (allEqual) {
			const sa = new ShapedArray(shape$1, dtype, weakType);
			return fullInternal(sa, data[0], device);
		}
	}
	const backend = require_backend.getBackend(device);
	const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	const slot = backend.malloc(data.byteLength, buf);
	return new Array$1({
		source: slot,
		st: require_backend.ShapeTracker.fromShape(shape$1),
		dtype,
		weakType,
		backend,
		committed: device != void 0
	});
}
function dataToJs(dtype, data, shape$1) {
	if (shape$1.length === 0) return dtype === require_backend.DType.Bool ? Boolean(data[0]) : data[0];
	const [first, ...rest] = shape$1;
	const restSize = require_backend.prod(rest);
	const ret = [];
	for (let i = 0; i < first; i++) {
		const subarray = data.slice(i * restSize, (i + 1) * restSize);
		ret.push(dataToJs(dtype, subarray, rest));
	}
	return ret;
}
/** If x is a value, lift it into an array, otherwise leave it be. */
function pureArray(x) {
	if (x instanceof Tracer) return x;
	else return array(x);
}
var EvalTrace = class extends Trace {
	pure = (x) => pureArray(x);
	lift = (x) => x;
	processPrimitive(primitive, tracers, params) {
		return implRules[primitive](tracers, params);
	}
};
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));
const implRules = Array$1._implRules();
function fullInternal(aval, fillValue, device) {
	return new Array$1({
		source: require_backend.AluExp.const(aval.dtype, fillValue),
		st: require_backend.ShapeTracker.fromShape(aval.shape),
		dtype: aval.dtype,
		weakType: aval.weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
function zerosLike$1(val, dtype) {
	return fullLike(val, 0, dtype);
}
function onesLike$1(val, dtype) {
	return fullLike(val, 1, dtype);
}
function fullLike(val, fillValue, dtype) {
	const aval = getAval(val);
	if (val instanceof Tracer) val.dispose();
	if (fillValue instanceof Tracer) throw new Error("numpy.fullLike() with array argument not implemented yet");
	const sa = new ShapedArray(aval.shape, dtype ?? aval.dtype, aval.weakType);
	return fullInternal(sa, fillValue);
}
/** Return a new array of given shape and type, filled with zeros. */
function zeros(shape$1, { dtype, device } = {}) {
	return full(shape$1, 0, {
		dtype,
		device
	});
}
/** Return a new array of given shape and type, filled with ones. */
function ones(shape$1, { dtype, device } = {}) {
	return full(shape$1, 1, {
		dtype,
		device
	});
}
/** Return a new array of given shape and type, filled with `fill_value`. */
function full(shape$1, fillValue, { dtype, device } = {}) {
	let weakType = dtype == void 0 && shape$1.length === 0;
	if (typeof fillValue === "number") dtype = dtype ?? require_backend.DType.Float32;
	else if (typeof fillValue === "boolean") {
		dtype = dtype ?? require_backend.DType.Bool;
		weakType = false;
	} else if (fillValue instanceof Tracer) throw new Error("numpy.full() with array argument not implemented yet");
	else throw new TypeError(`Invalid type for full: ${fillValue}`);
	return fullInternal(new ShapedArray(shape$1, dtype, weakType), fillValue, device);
}
/**
* Create an identity matrix.
*
* If numCols is not provided, it defaults to numRows, i.e., a square identity
* matrix with ones on the diagonal.
*/
function eye(numRows, numCols, { dtype, device } = {}) {
	numCols = numCols ?? numRows;
	const weakType = dtype == void 0;
	dtype = dtype ?? require_backend.DType.Float32;
	if (numCols < numRows) {
		const arr = eye(numCols, numRows, {
			dtype,
			device
		});
		return arr.transpose();
	}
	if (numRows === 0) return zeros([0, numCols], {
		dtype,
		device
	});
	const exp$2 = require_backend.AluExp.cmplt(require_backend.AluExp.mod(require_backend.AluVar.idx, require_backend.AluExp.i32(numCols + 1)), require_backend.AluExp.i32(1));
	return new Array$1({
		source: require_backend.AluExp.cast(dtype, exp$2),
		st: require_backend.ShapeTracker.fromShape([numRows, numCols]),
		dtype,
		weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/** Return the identity matrix, with ones on the main diagonal. */
function identity$1(n, { dtype, device } = {}) {
	return eye(n, n, {
		dtype,
		device
	});
}
/**
* Return evenly spaced values within a given interval.
*
* This can be called with a varying number of arguments, just like the range()
* builtin function in Python.
*
* - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
* - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
* - `arange(start, stop, step)` creates an array starting at `start`, ending
*   before `stop`, with a step size of `step`.
*
* Defaults to an integer data type. This can produce unintended results when
* using a non-integer step, so prefer linspace() in those cases.
*/
function arange(start, stop, step = 1, { dtype, device } = {}) {
	dtype = dtype ?? require_backend.DType.Int32;
	if (stop === void 0) {
		stop = start;
		start = 0;
	}
	if (step === 0) throw new RangeError(`Invalid step for arange: ${step}. Step must be non-zero.`);
	const size$1 = Math.max(0, Math.ceil((stop - start) / step));
	if (size$1 === 0) return zeros([0], {
		dtype,
		device
	});
	const exp$2 = require_backend.AluExp.add(require_backend.AluExp.const(dtype, start), require_backend.AluExp.mul(require_backend.AluExp.cast(dtype, require_backend.AluVar.idx), require_backend.AluExp.const(dtype, step)));
	const st = require_backend.ShapeTracker.fromShape([size$1]);
	return new Array$1({
		source: exp$2,
		st,
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/**
* Return an array with ones on and below the diagonal and zeros elsewhere.
*
* If `k` is provided, it specifies the sub-diagonal on and below which the
* array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
* `k>0` is above it.
*/
function tri(n, m, k = 0, { dtype, device } = {}) {
	m ??= n;
	dtype ??= require_backend.DType.Float32;
	if (!Number.isInteger(n) || n < 0) throw new Error(`tri: n must be a non-negative integer, got ${n}`);
	if (!Number.isInteger(m) || m < 0) throw new Error(`tri: m must be a non-negative integer, got ${m}`);
	if (!Number.isInteger(k)) throw new Error(`tri: k must be an integer, got ${k}`);
	const rows = arange(k, n + k, 1, {
		dtype: require_backend.DType.Int32,
		device
	});
	const cols = arange(0, m, 1, {
		dtype: require_backend.DType.Int32,
		device
	});
	return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}
/** Return the lower triangle of an array. Must be of dimension >= 2. */
function tril(a, k = 0) {
	if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
	a = fudgeArray(a);
	const [n, m] = a.shape.slice(-2);
	return where$1(tri(n, m, k, { dtype: require_backend.DType.Bool }), a.ref, zerosLike$1(a));
}
/** Return the upper triangle of an array. Must be of dimension >= 2. */
function triu(a, k = 0) {
	if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
	a = fudgeArray(a);
	const [n, m] = a.shape.slice(-2);
	return where$1(tri(n, m, k - 1, { dtype: require_backend.DType.Bool }), zerosLike$1(a.ref), a);
}
/**
* Return evenly spaced numbers over a specified interval.
*
* Returns _num_ evenly spaced samples, calculated over the interval
* [`start`, `stop`]. The endpoint `stop` is included in the result by default,
* but this is controlled by the `endpoint` parameter.
*
* The default data type is Float32. Use arange() for integer steps.
*/
function linspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
	dtype = dtype ?? require_backend.DType.Float32;
	if (num < 0 || !Number.isInteger(num)) throw new RangeError(`Invalid num for linspace: ${num}. Must be non-negative integer.`);
	else if (num === 0) return zeros([0], {
		dtype,
		device
	});
	else if (num === 1) return full([1], start, {
		dtype,
		device
	});
	else if (start === stop) return full([num], start, {
		dtype,
		device
	});
	const delta = stop - start;
	const denom = endpoint ? num - 1 : num;
	const exp$2 = require_backend.AluExp.cast(dtype, require_backend.AluExp.add(require_backend.AluExp.f32(start), require_backend.AluExp.mul(require_backend.AluExp.f32(delta / denom), require_backend.AluExp.cast(require_backend.DType.Float32, require_backend.AluVar.idx))));
	const st = require_backend.ShapeTracker.fromShape([num]);
	return new Array$1({
		source: exp$2,
		st,
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/**
* Return numbers spaced evenly on a log scale.
*
* In linear space, the sequence starts at `base ** start` and ends at
* `base ** stop` (see `endpoint` below).
*
* @param start - `base ** start` is the starting value of the sequence.
* @param stop - `base ** stop` is the final value of the sequence, unless `endpoint` is false.
* @param num - Number of samples to generate. Default is 50.
* @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
* @param base - The base of the log space. Default is 10.
* @returns Array of evenly spaced values on a log scale.
*/
function logspace(start, stop, num = 50, endpoint = true, base = 10, { dtype, device } = {}) {
	const y = linspace(start, stop, num, endpoint, {
		dtype,
		device
	});
	const logBase = Math.log(base);
	return exp$1(mul(y, logBase));
}
function aluCompare(a, b, op) {
	switch (op) {
		case CompareOp.Less: return require_backend.AluExp.cmplt(a, b);
		case CompareOp.Equal: return require_backend.AluExp.cmpne(a, b).not();
		case CompareOp.NotEqual: return require_backend.AluExp.cmpne(a, b);
		case CompareOp.LessEqual: return require_backend.AluExp.add(require_backend.AluExp.cmplt(a, b), require_backend.AluExp.cmpne(a, b).not());
	}
}

//#endregion
//#region src/frontend/vmap.ts
var import_usingCtx$1 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
function mappedAval(batchDim, aval) {
	const shape$1 = [...aval.shape];
	shape$1.splice(batchDim, 1);
	return new ShapedArray(shape$1, aval.dtype, aval.weakType);
}
/** Move one axis to a different index. */
function moveaxis(x, src, dst) {
	const t = pureArray(x);
	src = require_backend.checkAxis(src, t.ndim);
	dst = require_backend.checkAxis(dst, t.ndim);
	if (src === dst) return t;
	const perm = require_backend.range(t.ndim);
	perm.splice(src, 1);
	perm.splice(dst, 0, src);
	return transpose$1(t, perm);
}
function moveBatchAxis(axisSize, src, dst, x) {
	if (src === null) {
		const targetShape = [...x.shape];
		targetShape.splice(dst, 0, axisSize);
		return broadcast(x, targetShape, [dst]);
	} else if (src === dst) return x;
	else return moveaxis(x, src, dst);
}
var BatchTracer = class extends Tracer {
	constructor(trace$1, val, batchDim) {
		super(trace$1);
		this.val = val;
		this.batchDim = batchDim;
	}
	get aval() {
		if (this.batchDim === null) return this.val.aval;
		else return mappedAval(this.batchDim, this.val.aval);
	}
	toString() {
		return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
	}
	get ref() {
		this.val.ref;
		return this;
	}
	dispose() {
		this.val.dispose();
	}
	fullLower() {
		if (this.batchDim === null) return this.val.fullLower();
		else return this;
	}
};
var BatchTrace = class extends Trace {
	pure(val) {
		return this.lift(pureArray(val));
	}
	lift(val) {
		return new BatchTracer(this, val, null);
	}
	processPrimitive(primitive, tracers, params) {
		const [valsIn, bdimsIn] = require_backend.unzip2(tracers.map((t) => [t.val, t.batchDim]));
		const vmapRule = vmapRules[primitive];
		if (vmapRule === void 0) throw new Error(`No vmap rule for: ${primitive}`);
		if (bdimsIn.every((d) => d === null)) {
			const valOuts$1 = bind(primitive, valsIn, params);
			return valOuts$1.map((x) => new BatchTracer(this, x, null));
		}
		const [valOuts, bdimOuts] = vmapRule(this.axisSize, valsIn, bdimsIn, params);
		if (valOuts.length !== bdimOuts.length) throw new Error(`vmap rule for ${primitive} returned mismatched lengths: ${valOuts.length} vs ${bdimOuts.length}`);
		return require_backend.zip(valOuts, bdimOuts).map(([x, bd]) => new BatchTracer(this, x, bd));
	}
	get axisSize() {
		return this.main.globalData;
	}
};
/**
* Process a primitive with built-in broadcasting.
*
* Reference: https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/interpreters/batching.py#L1029
*/
function broadcastBatcher(prim) {
	return (axisSize, args, dims, params) => {
		if (args.length === 0) throw new Error("Empty list in broadcastBatcher");
		const nd = Math.max(...args.map((x, i) => ndim$1(x) + (dims[i] === null ? 1 : 0)));
		const firstIdx = dims.findIndex((d) => d !== null);
		const firstBdim = dims[firstIdx] - args[firstIdx].ndim;
		if (require_backend.zip(args, dims).every(([x, d]) => d === null && ndim$1(x) < -firstBdim || d !== null && d - x.ndim === firstBdim)) return [[bind1(prim, args, params)], [nd + firstBdim]];
		args = args.map((x, i) => {
			if (dims[i] === null) return x;
			x = moveBatchAxis(axisSize, dims[i], 0, x);
			if (x.ndim < nd) x = x.reshape([
				x.shape[0],
				...require_backend.rep(nd - x.ndim, 1),
				...x.shape.slice(1)
			]);
			return x;
		});
		return [[bind1(prim, args, params)], [0]];
	};
}
function unopBatcher(prim) {
	return (axisSize, [x], [xBdim], params) => {
		return [[bind1(prim, [x], params)], [xBdim]];
	};
}
function lastDimsBatcher(prim, inputDims, numOutputs = 1) {
	return (axisSize, [x], [xBdim], params) => {
		require_backend.assertNonNull(xBdim);
		if (xBdim < x.ndim - inputDims) return [bind(prim, [x], params), require_backend.rep(numOutputs, xBdim)];
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		return [bind(prim, [x], params), require_backend.rep(numOutputs, 0)];
	};
}
const vmapRules = {
	[Primitive.Add]: broadcastBatcher(Primitive.Add),
	[Primitive.Mul]: broadcastBatcher(Primitive.Mul),
	[Primitive.Idiv]: broadcastBatcher(Primitive.Idiv),
	[Primitive.Mod]: broadcastBatcher(Primitive.Mod),
	[Primitive.Min]: broadcastBatcher(Primitive.Min),
	[Primitive.Max]: broadcastBatcher(Primitive.Max),
	[Primitive.Neg]: unopBatcher(Primitive.Neg),
	[Primitive.Reciprocal]: unopBatcher(Primitive.Reciprocal),
	[Primitive.Floor]: unopBatcher(Primitive.Floor),
	[Primitive.Ceil]: unopBatcher(Primitive.Ceil),
	[Primitive.StopGradient]: unopBatcher(Primitive.StopGradient),
	[Primitive.Cast]: unopBatcher(Primitive.Cast),
	[Primitive.Bitcast]: unopBatcher(Primitive.Bitcast),
	[Primitive.Sin]: unopBatcher(Primitive.Sin),
	[Primitive.Cos]: unopBatcher(Primitive.Cos),
	[Primitive.Asin]: unopBatcher(Primitive.Asin),
	[Primitive.Atan]: unopBatcher(Primitive.Atan),
	[Primitive.Exp]: unopBatcher(Primitive.Exp),
	[Primitive.Log]: unopBatcher(Primitive.Log),
	[Primitive.Erf]: unopBatcher(Primitive.Erf),
	[Primitive.Erfc]: unopBatcher(Primitive.Erfc),
	[Primitive.Sqrt]: unopBatcher(Primitive.Sqrt),
	[Primitive.Reduce](axisSize, [x], [xBdim], { op, axis }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
		return [[reduce(x, op, newAxis)], [outBdim]];
	},
	[Primitive.Dot](axisSize, [x, y], [xBdim, yBdim]) {
		x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
		y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
		const z = dot$2(x, y);
		return [[z], [z.ndim - 1]];
	},
	[Primitive.Conv](axisSize, [x, y], [xBdim, yBdim], params) {
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		y = moveBatchAxis(axisSize, yBdim, 0, y);
		const z = conv$1(x, y, {
			...params,
			vmapDims: params.vmapDims + 1
		});
		return [[z], [0]];
	},
	[Primitive.Compare]: broadcastBatcher(Primitive.Compare),
	[Primitive.Where]: broadcastBatcher(Primitive.Where),
	[Primitive.Concatenate](axisSize, xs, xBdims, { axis }) {
		const minBdim = Math.min(...xBdims.filter((d) => d !== null));
		xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
		const newAxis = axis + (minBdim <= axis ? 1 : 0);
		return [[concatenate$1(xs, newAxis)], [minBdim]];
	},
	[Primitive.Split](axisSize, [x], [xBdim], { axis, sizes }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis + (xBdim <= axis ? 1 : 0);
		const outs = split$2(x, newAxis, sizes);
		return [outs, require_backend.rep(outs.length, xBdim)];
	},
	[Primitive.RandomBits](axisSize, [k0, k1], [bdim0, bdim1], { shape: shape$1, mode }) {
		k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
		k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
		return [[randomBits(k0, k1, [axisSize, ...shape$1], mode)], [0]];
	},
	[Primitive.Gather](axisSize, [x, ...indices], [xBdim, ...indicesBdim], { axis, outDim }) {
		if (indicesBdim.every((d) => d === null)) {
			require_backend.assertNonNull(xBdim);
			const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
			let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
			let newOutDim = outDim;
			if (newOutDim < newBdim) newBdim += axis.length;
			else newOutDim += 1;
			return [[gather(x, indices, newAxis, newOutDim)], [newBdim]];
		}
		const nd = Math.max(...indices.map((m, i) => ndim$1(m) + (indicesBdim[i] === null ? 1 : 0)));
		indices = indices.map((m, i) => {
			if (indicesBdim[i] === null) return m;
			m = moveBatchAxis(axisSize, indicesBdim[i], 0, m);
			if (m.ndim < nd) m = m.reshape([
				m.shape[0],
				...require_backend.rep(nd - m.ndim, 1),
				...m.shape.slice(1)
			]);
			return m;
		});
		if (xBdim === null) return [[gather(x, indices, axis, outDim)], [outDim]];
		else {
			x = moveBatchAxis(axisSize, xBdim, 0, x);
			const newAxis = [0, ...axis.map((ax) => ax + 1)];
			const extraBatchIndex = arange(axisSize).reshape([-1, ...require_backend.rep(nd - 1, 1)]);
			indices.splice(0, 0, extraBatchIndex);
			return [[gather(x, indices, newAxis, outDim)], [outDim]];
		}
	},
	[Primitive.Transpose](axisSize, [x], [xBdim], { perm }) {
		require_backend.assertNonNull(xBdim);
		const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
		newPerm.splice(xBdim, 0, xBdim);
		return [[transpose$1(x, newPerm)], [xBdim]];
	},
	[Primitive.Broadcast](axisSize, [x], [xBdim], { shape: shape$1, axis }) {
		require_backend.assertNonNull(xBdim);
		const newShape = shape$1.toSpliced(xBdim, 0, axisSize);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		return [[broadcast(x, newShape, newAxis)], [xBdim]];
	},
	[Primitive.Reshape](axisSize, [x], [xBdim], { shape: shape$1 }) {
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		return [[reshape$1(x, [axisSize, ...shape$1])], [0]];
	},
	[Primitive.Flip](axisSize, [x], [xBdim], { axis }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		return [[flip$1(x, newAxis)], [xBdim]];
	},
	[Primitive.Shrink](axisSize, [x], [xBdim], { slice }) {
		require_backend.assertNonNull(xBdim);
		const newSlice = slice.toSpliced(xBdim, 0, [0, axisSize]);
		return [[shrink(x, newSlice)], [xBdim]];
	},
	[Primitive.Pad](axisSize, [x], [xBdim], { width }) {
		require_backend.assertNonNull(xBdim);
		const newWidth = width.toSpliced(xBdim, 0, [0, 0]);
		return [[pad$1(x, newWidth)], [xBdim]];
	},
	[Primitive.Sort]: lastDimsBatcher(Primitive.Sort, 1),
	[Primitive.Argsort]: lastDimsBatcher(Primitive.Argsort, 1, 2),
	[Primitive.TriangularSolve](axisSize, [a, b], [aBdim, bBdim], { unitDiagonal }) {
		if (aBdim === null) {
			b = moveBatchAxis(axisSize, bBdim, -3, b);
			const [s, m, n] = b.shape.slice(-3);
			b = b.reshape([
				...b.shape.slice(0, -3),
				s * m,
				n
			]);
			let x$1 = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
			x$1 = x$1.reshape([
				...b.shape.slice(0, -2),
				s,
				m,
				n
			]);
			return [[x$1], [x$1.ndim - 3]];
		}
		a = moveBatchAxis(axisSize, aBdim, 0, a);
		b = moveBatchAxis(axisSize, bBdim, 0, b);
		const x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
		return [[x], [0]];
	},
	[Primitive.Cholesky]: lastDimsBatcher(Primitive.Cholesky, 2),
	[Primitive.LU]: lastDimsBatcher(Primitive.LU, 2, 3),
	[Primitive.Jit](axisSize, args, dims, { name, jaxpr }) {
		const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
		const outs = bind(Primitive.Jit, [...newJaxpr.consts.map((c) => c.ref), ...args], {
			name: `${name}_vmap`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		return [outs, require_backend.rep(outs.length, 0)];
	},
	[Primitive.Scan](axisSize, args, dims, { jaxpr, numCarry, numConsts, length, reverse }) {
		const numX = args.length - numConsts - numCarry;
		const numY = jaxpr.outs.length - numCarry;
		const consts = args.slice(0, numConsts);
		const initCarry = args.slice(numConsts, numConsts + numCarry);
		const xs = args.slice(numConsts + numCarry);
		const constDims = dims.slice(0, numConsts);
		const carryDims = dims.slice(numConsts, numConsts + numCarry);
		const xsDims = dims.slice(numConsts + numCarry);
		const movedConsts = consts.map((c, i) => moveBatchAxis(axisSize, constDims[i], 0, c));
		const movedCarry = initCarry.map((c, i) => moveBatchAxis(axisSize, carryDims[i], 0, c));
		const movedXs = xs.map((x, i) => {
			if (xsDims[i] === null) {
				const newShape = [
					x.shape[0],
					axisSize,
					...x.shape.slice(1)
				];
				return broadcast(x, newShape, [1]);
			} else if (xsDims[i] === 0) return moveaxis(x, 0, 1);
			else return moveBatchAxis(axisSize, xsDims[i], 1, x);
		});
		const bodyDims = [
			...require_backend.rep(numConsts, 0),
			...require_backend.rep(numCarry, 0),
			...require_backend.rep(numX, 0)
		];
		const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);
		const scanArgs = [
			...vmappedBody.consts.map((c) => c.ref),
			...movedConsts,
			...movedCarry,
			...movedXs
		];
		const results = bind(Primitive.Scan, scanArgs, {
			jaxpr: vmappedBody.jaxpr,
			numCarry,
			numConsts: vmappedBody.consts.length,
			length,
			reverse
		});
		const carryOut = results.slice(0, numCarry);
		const ysOut = results.slice(numCarry);
		const movedYs = ysOut.map((y) => moveaxis(y, 1, 0));
		return [[...carryOut, ...movedYs], require_backend.rep(numCarry + numY, 0)];
	}
};
const vmapJaxprCache = /* @__PURE__ */ new Map();
function vmapJaxpr(jaxpr, axisSize, dims) {
	const cacheKey = JSON.stringify([axisSize, dims]);
	const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const inAvals = jaxpr.inBinders.map((v, i) => {
		if (dims[i] === null) return v.aval;
		const shape$1 = [...v.aval.shape];
		shape$1.splice(dims[i], 0, axisSize);
		return new ShapedArray(shape$1, v.aval.dtype, v.aval.weakType);
	});
	const { jaxpr: newJaxpr } = makeJaxpr$1((args) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(inAvals);
	if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	vmapJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vmapFlat(f, inAxes, args) {
	let axisSize = void 0;
	for (let i = 0; i < args.length; i++) if (inAxes[i] !== null) {
		const arg = args[i];
		if (!(arg instanceof Tracer)) throw new TypeError("vmap requires Tracer argument for mapped axes");
		const size$1 = arg.shape[inAxes[i]];
		if (axisSize === void 0) axisSize = size$1;
		else if (axisSize !== size$1) throw new TypeError("vmap requires all mapped axes to have the same size");
	}
	if (axisSize === void 0) throw new TypeError("vmap requires at least one mapped axis");
	let valsOut, bdimsOut;
	try {
		var _usingCtx$1 = (0, import_usingCtx$1.default)();
		const main = _usingCtx$1.u(newMain(BatchTrace, axisSize));
		const trace$1 = new BatchTrace(main);
		const tracersIn = args.map((x, i) => inAxes[i] === null ? pureArray(x) : new BatchTracer(trace$1, pureArray(x), inAxes[i]));
		const outs = f(...tracersIn);
		const tracersOut = outs.map((out) => fullRaise(trace$1, out));
		[valsOut, bdimsOut] = require_backend.unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
	return require_backend.zip(valsOut, bdimsOut).map(([valOut, bdim]) => moveBatchAxis(axisSize, bdim, 0, valOut));
}
function vmap$1(f, inAxes = 0) {
	return (...args) => {
		const [argsFlat, inTree] = flatten(args);
		let inAxesFlat = [];
		if (typeof inAxes === "number") inAxesFlat = require_backend.rep(argsFlat.length, inAxes);
		else for (let i = 0; i < args.length; i++) if (inAxes[i] == null) inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, null));
		else if (typeof inAxes[i] === "number") inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, inAxes[i]));
		else {
			const [axesFlat, axesTreeDef] = flatten(inAxes[i]);
			if (!inTree.childTreedefs[i].equals(axesTreeDef)) throw new TreeMismatchError("vmap", inTree.childTreedefs[i], axesTreeDef);
			inAxesFlat.push(...axesFlat);
		}
		const [fFlat, outTree] = flattenFun(f, inTree);
		const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
		if (outTree.value === void 0) throw new Error("outTree was not set in vmap");
		return unflatten(outTree.value, outsFlat);
	};
}
function jacfwd$1(f) {
	return function jacobianForward(x) {
		if (x.shape.length !== 1) throw new TypeError("jacfwd only supports 1D inputs");
		const [size$1] = x.shape;
		const pushfwd = (v) => jvp$1(f, [x], [v])[1];
		return vmap$1(pushfwd, [0])(eye(size$1, void 0, { dtype: x.dtype }));
	};
}

//#endregion
//#region src/frontend/jvp.ts
var import_usingCtx = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
var JVPTracer = class extends Tracer {
	constructor(trace$1, primal, tangent) {
		super(trace$1);
		this.primal = primal;
		this.tangent = tangent;
	}
	get aval() {
		return this.primal.aval;
	}
	toString() {
		return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
	}
	get ref() {
		this.primal.ref, this.tangent.ref;
		return this;
	}
	dispose() {
		this.primal.dispose();
		this.tangent.dispose();
	}
};
var JVPTrace = class extends Trace {
	pure(val) {
		return this.lift(pureArray(val));
	}
	lift(val) {
		return new JVPTracer(this, val, zerosLike$1(val.ref));
	}
	processPrimitive(primitive, tracers, params) {
		const [primalsIn, tangentsIn] = require_backend.unzip2(tracers.map((x) => [x.primal, x.tangent]));
		const jvpRule = jvpRules[primitive];
		if (jvpRule === void 0) throw new Error(`No JVP rule for: ${primitive}`);
		const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
		return require_backend.zip(primalsOut, tangentsOut).map(([x, t]) => new JVPTracer(this, x, t));
	}
};
/** Rule that applies the same operation to primals and tangents. */
function linearTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		const ys = bind(primitive, primals, params);
		const dys = bind(primitive, tangents, params);
		return [ys, dys];
	};
}
/** Rule for product of gradients in bilinear operations. */
function bilinearTangentsJvp(primitive) {
	return ([x, y], [dx, dy], params) => {
		const primal = bind1(primitive, [x.ref, y.ref], params);
		const tangent = bind1(primitive, [x, dy], params).add(bind1(primitive, [dx, y], params));
		return [[primal], [tangent]];
	};
}
/** Rule that zeros out any tangents. */
function zeroTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		for (const t of tangents) t.dispose();
		const ys = bind(primitive, primals, params);
		return [ys, ys.map((y) => zerosLike$1(y.ref))];
	};
}
/** Compute `a @ b.T`, batched to last two axes. */
function batchMatmulT(a, b) {
	return dot$2(a.reshape(a.shape.toSpliced(-1, 0, 1)), b.reshape(b.shape.toSpliced(-2, 0, 1)));
}
/** Batch matrix transpose. */
function mT(a) {
	return moveaxis(a, -2, -1);
}
function sliceAxis(a, axis, p) {
	const slices = Array(a.shape.length).fill([]);
	slices[require_backend.checkAxis(axis, a.ndim)] = p;
	return a.slice(...slices);
}
function padAxis(a, axis, p) {
	const pads = Array(a.shape.length).fill([0, 0]);
	pads[require_backend.checkAxis(axis, a.ndim)] = p;
	return pad$1(a, pads);
}
const jvpRules = {
	[Primitive.Add]: linearTangentsJvp(Primitive.Add),
	[Primitive.Mul]: bilinearTangentsJvp(Primitive.Mul),
	[Primitive.Idiv]: zeroTangentsJvp(Primitive.Idiv),
	[Primitive.Mod]([x, y], [dx, dy]) {
		if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) {
			dx.dispose();
			dy.dispose();
			return [[x.ref, y.ref], [zerosLike$1(x), zerosLike$1(y)]];
		}
		const q = idiv(x.ref, y.ref);
		return [[mod(x, y)], [dx.sub(dy.mul(q))]];
	},
	[Primitive.Min]([x, y], [dx, dy]) {
		return [[min$1(x.ref, y.ref)], [where$1(less$1(y, x), dy, dx)]];
	},
	[Primitive.Max]([x, y], [dx, dy]) {
		return [[max$1(x.ref, y.ref)], [where$1(less$1(x, y), dy, dx)]];
	},
	[Primitive.Neg]: linearTangentsJvp(Primitive.Neg),
	[Primitive.Reciprocal]([x], [dx]) {
		const xRecip = reciprocal$1(x.ref);
		return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
	},
	[Primitive.Floor]: zeroTangentsJvp(Primitive.Floor),
	[Primitive.Ceil]: zeroTangentsJvp(Primitive.Ceil),
	[Primitive.StopGradient]: zeroTangentsJvp(Primitive.StopGradient),
	[Primitive.Cast]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		if (require_backend.isFloatDtype(dtype) && require_backend.isFloatDtype(x.dtype)) return [[cast(x, dtype)], [cast(dx, dtype)]];
		else {
			dx.dispose();
			return [[cast(x.ref, dtype)], [zerosLike$1(x)]];
		}
	},
	[Primitive.Bitcast]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		dx.dispose();
		return [[bitcast(x.ref, dtype)], [zerosLike$1(x)]];
	},
	[Primitive.Sin]([x], [dx]) {
		return [[sin$1(x.ref)], [cos$1(x).mul(dx)]];
	},
	[Primitive.Cos]([x], [dx]) {
		return [[cos$1(x.ref)], [neg(sin$1(x)).mul(dx)]];
	},
	[Primitive.Asin]([x], [dx]) {
		const denom = sqrt$1(reciprocal$1(cast(1, x.dtype).sub(x.ref.mul(x.ref))));
		return [[asin$1(x)], [denom.mul(dx)]];
	},
	[Primitive.Atan]([x], [dx]) {
		const denom = cast(1, x.dtype).add(x.ref.mul(x.ref));
		return [[atan$1(x)], [dx.div(denom)]];
	},
	[Primitive.Exp]([x], [dx]) {
		const z = exp$1(x);
		return [[z.ref], [z.mul(dx)]];
	},
	[Primitive.Log]([x], [dx]) {
		return [[log$1(x.ref)], [reciprocal$1(x).mul(dx)]];
	},
	[Primitive.Erf]([x], [dx]) {
		const coeff = 2 / Math.sqrt(Math.PI);
		const expTerm = exp$1(neg(x.ref.mul(x.ref)));
		return [[erf$1(x)], [expTerm.mul(coeff).mul(dx)]];
	},
	[Primitive.Erfc]([x], [dx]) {
		const coeff = -2 / Math.sqrt(Math.PI);
		const expTerm = exp$1(neg(x.ref.mul(x.ref)));
		return [[erfc$1(x)], [expTerm.mul(coeff).mul(dx)]];
	},
	[Primitive.Sqrt]([x], [dx]) {
		const z = sqrt$1(x);
		return [[z.ref], [reciprocal$1(z.mul(2)).mul(dx)]];
	},
	[Primitive.Reduce]([x], [dx], { op, axis }) {
		if (op === require_backend.AluOp.Add) return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
		else if (op === require_backend.AluOp.Mul) {
			const primal = reduce(x.ref, op, axis);
			const tangent = broadcast(primal.ref, x.shape, axis).mul(reciprocal$1(x)).mul(dx).sum(axis);
			return [[primal], [tangent]];
		} else if (op === require_backend.AluOp.Min || op === require_backend.AluOp.Max) {
			const primal = reduce(x.ref, op, axis);
			const notMin = notEqual$1(x, broadcast(primal.ref, x.shape, axis));
			const minCount = where$1(notMin.ref, 0, 1).sum(axis);
			const tangent = where$1(notMin, 0, dx).sum(axis).div(minCount);
			return [[primal], [tangent]];
		} else throw new Error(`JVP rule not implemented for reduce op: ${op}`);
	},
	[Primitive.Pool]: linearTangentsJvp(Primitive.Pool),
	[Primitive.PoolTranspose]: linearTangentsJvp(Primitive.PoolTranspose),
	[Primitive.Dot]: bilinearTangentsJvp(Primitive.Dot),
	[Primitive.Conv]: bilinearTangentsJvp(Primitive.Conv),
	[Primitive.Compare]: zeroTangentsJvp(Primitive.Compare),
	[Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
		dcond.dispose();
		return [[where$1(cond.ref, x, y)], [where$1(cond, dx, dy)]];
	},
	[Primitive.Concatenate]: linearTangentsJvp(Primitive.Concatenate),
	[Primitive.Split]: linearTangentsJvp(Primitive.Split),
	[Primitive.RandomBits]: zeroTangentsJvp(Primitive.RandomBits),
	[Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
		const indicesRef = indices.map((t) => t.ref);
		return [[gather(x, indices, axis, outDim)], [gather(dx, indicesRef, axis, outDim)]];
	},
	[Primitive.Transpose]: linearTangentsJvp(Primitive.Transpose),
	[Primitive.Broadcast]: linearTangentsJvp(Primitive.Broadcast),
	[Primitive.Reshape]: linearTangentsJvp(Primitive.Reshape),
	[Primitive.Flip]: linearTangentsJvp(Primitive.Flip),
	[Primitive.Shrink]: linearTangentsJvp(Primitive.Shrink),
	[Primitive.Pad]: linearTangentsJvp(Primitive.Pad),
	[Primitive.Sort]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		return [[y], [gather(dx, [idx], [-1], -1)]];
	},
	[Primitive.Argsort]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		return [[y, idx.ref], [gather(dx, [idx.ref], [-1], -1), zerosLike$1(idx)]];
	},
	[Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
		const x = triangularSolve$1(a.ref, b, { unitDiagonal });
		const dax = batchMatmulT(da, x.ref);
		const rhsT = db.sub(mT(dax));
		const dx = triangularSolve$1(a, rhsT, { unitDiagonal });
		return [[x], [dx]];
	},
	[Primitive.Cholesky]([a], [da]) {
		const L = cholesky$2(a.ref);
		da = da.ref.add(mT(da)).mul(.5);
		const W = triangularSolve$1(L.ref, da, { lower: true });
		const ST = triangularSolve$1(L.ref, mT(W), { lower: true });
		const dL = batchMatmulT(L.ref, triu(ST.ref, 1).add(triu(ST)).mul(.5));
		return [[L], [dL]];
	},
	[Primitive.LU]([a], [da]) {
		const [luMatrix, pivots, permutation] = lu$1(a);
		const [m, n] = a.shape.slice(-2);
		const k = Math.min(m, n);
		const luSliceL = sliceAxis(luMatrix.ref, -1, [0, k]);
		const lLower = tril(luSliceL, -1);
		const lPadded = m > k ? padAxis(lLower, -1, [0, m - k]) : lLower;
		const L = lPadded.add(eye(m));
		const luSliceU = sliceAxis(luMatrix.ref, -2, [0, k]);
		const uUpper = triu(luSliceU);
		const uPadded = n > k ? padAxis(uUpper, -2, [0, n - k]) : uUpper;
		const uEye = n > k ? padAxis(padAxis(eye(n - k), -1, [k, 0]), -2, [k, 0]) : zerosLike$1(uPadded.ref);
		const U = uPadded.add(uEye);
		const P = permutation.ref.reshape([...permutation.shape, 1]).equal(arange(m)).astype(da.dtype);
		const pda = batchMatmulT(P, mT(da));
		const la = mT(triangularSolve$1(L.ref, mT(pda), {
			lower: true,
			unitDiagonal: true
		}));
		const lau = triangularSolve$1(mT(U.ref), la, { lower: true });
		const lDot = batchMatmulT(L, mT(tril(lau.ref, -1)));
		const uDot = batchMatmulT(triu(lau), mT(U));
		return [[
			luMatrix,
			pivots,
			permutation
		], [
			lDot.add(uDot),
			zerosLike$1(pivots.ref),
			zerosLike$1(permutation.ref)
		]];
	},
	[Primitive.Jit](primals, tangents, { name, jaxpr }) {
		const newJaxpr = jvpJaxpr(jaxpr);
		const outs = bind(Primitive.Jit, [
			...newJaxpr.consts.map((c) => c.ref),
			...primals,
			...tangents
		], {
			name: `${name}_jvp`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		const n = outs.length / 2;
		if (!Number.isInteger(n)) throw new Error("internal: JVP Jaxpr output length is not even");
		const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
		return [primalsOut, tangentsOut];
	},
	[Primitive.Scan](primals, tangents, { jaxpr, numCarry, numConsts, length, reverse }) {
		const numX = primals.length - numConsts - numCarry;
		const numY = jaxpr.outs.length - numCarry;
		const jvpBody = jvpJaxpr(jaxpr);
		const numJvpConsts = jvpBody.consts.length;
		const numBodyInputs = numConsts + numCarry + numX;
		const jvpOrderAvals = jvpBody.jaxpr.inBinders.slice(numJvpConsts).map((v) => v.aval);
		const constsP_avals = jvpOrderAvals.slice(0, numConsts);
		const carryP_avals = jvpOrderAvals.slice(numConsts, numConsts + numCarry);
		const xP_avals = jvpOrderAvals.slice(numConsts + numCarry, numBodyInputs);
		const constsT_avals = jvpOrderAvals.slice(numBodyInputs, numBodyInputs + numConsts);
		const carryT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts, numBodyInputs + numConsts + numCarry);
		const xT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numCarry);
		const wrapperInAvals = [
			...constsP_avals,
			...constsT_avals,
			...carryP_avals,
			...carryT_avals,
			...xP_avals,
			...xT_avals
		];
		const { jaxpr: wrapperJaxpr } = makeJaxpr$1((...scanOrderArgs) => {
			const constsP_in = scanOrderArgs.slice(0, numConsts);
			const constsT_in = scanOrderArgs.slice(numConsts, numConsts * 2);
			const carryP_in = scanOrderArgs.slice(numConsts * 2, numConsts * 2 + numCarry);
			const carryT_in = scanOrderArgs.slice(numConsts * 2 + numCarry, numConsts * 2 + numCarry * 2);
			const xP_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2, numConsts * 2 + numCarry * 2 + numX);
			const xT_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2 + numX);
			const jvpOrderArgs = [
				...constsP_in.map((x) => x.ref),
				...carryP_in.map((x) => x.ref),
				...xP_in.map((x) => x.ref),
				...constsT_in.map((x) => x.ref),
				...carryT_in.map((x) => x.ref),
				...xT_in.map((x) => x.ref)
			];
			const jvpOutputs = bind(Primitive.Jit, [...jvpBody.consts.map((c) => c.ref), ...jvpOrderArgs], {
				jaxpr: jvpBody.jaxpr,
				numConsts: numJvpConsts,
				name: "jvp_body"
			});
			const carryP_out = jvpOutputs.slice(0, numCarry);
			const yP_out = jvpOutputs.slice(numCarry, numCarry + numY);
			const carryT_out = jvpOutputs.slice(numCarry + numY, numCarry * 2 + numY);
			const yT_out = jvpOutputs.slice(numCarry * 2 + numY);
			return [
				...carryP_out,
				...carryT_out,
				...yP_out,
				...yT_out
			];
		})(...wrapperInAvals);
		const constsP = primals.slice(0, numConsts);
		const carryP = primals.slice(numConsts, numConsts + numCarry);
		const xsP = primals.slice(numConsts + numCarry);
		const constsT = tangents.slice(0, numConsts);
		const carryT = tangents.slice(numConsts, numConsts + numCarry);
		const xsT = tangents.slice(numConsts + numCarry);
		const scanArgsJvp = [
			...wrapperJaxpr.consts.map((c) => c.ref),
			...constsP.map((c) => c.ref),
			...constsT.map((c) => c.ref),
			...carryP.map((c) => c.ref),
			...carryT.map((c) => c.ref),
			...xsP.map((x) => x.ref),
			...xsT.map((x) => x.ref)
		];
		const results = bind(Primitive.Scan, scanArgsJvp, {
			jaxpr: wrapperJaxpr.jaxpr,
			numCarry: numCarry * 2,
			numConsts: wrapperJaxpr.consts.length + numConsts * 2,
			length,
			reverse
		});
		wrapperJaxpr.dispose();
		const carryOutP = results.slice(0, numCarry);
		const carryOutT = results.slice(numCarry, numCarry * 2);
		const ysP = results.slice(numCarry * 2, numCarry * 2 + numY);
		const ysT = results.slice(numCarry * 2 + numY);
		const primalsOut = [...carryOutP, ...ysP];
		const tangentsOut = [...carryOutT, ...ysT];
		return [primalsOut, tangentsOut];
	}
};
const jvpJaxprCache = /* @__PURE__ */ new Map();
function jvpJaxpr(jaxpr) {
	if (jvpJaxprCache.has(jaxpr)) return jvpJaxprCache.get(jaxpr);
	const inAvals = jaxpr.inBinders.map((v) => v.aval);
	const { jaxpr: newJaxpr } = makeJaxpr$1((primals, tangents) => jvpFlat(jaxprAsFun(jaxpr), primals, tangents))(inAvals, inAvals);
	jvpJaxprCache.set(jaxpr, newJaxpr);
	return newJaxpr;
}
function jvpFlat(f, primals, tangents) {
	try {
		var _usingCtx$1 = (0, import_usingCtx.default)();
		const main = _usingCtx$1.u(newMain(JVPTrace));
		const trace$1 = new JVPTrace(main);
		const tracersIn = require_backend.zip(primals, tangents).map(([x, t]) => new JVPTracer(trace$1, pureArray(x), pureArray(t)));
		const outs = f(...tracersIn);
		const tracersOut = outs.map((out) => fullRaise(trace$1, out));
		return require_backend.unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
function jvp$1(f, primals, tangents, { hasAux = false } = {}) {
	const [primalsFlat, inTree] = flatten(primals);
	const [tangentsFlat, inTree2] = flatten(tangents);
	if (!inTree.equals(inTree2)) throw new TreeMismatchError("jvp", inTree, inTree2);
	let flatFun, outTree, aux;
	if (hasAux) [flatFun, outTree, aux] = flattenFunWithAux(f, inTree);
	else [flatFun, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, tangentsOutFlat] = jvpFlat(flatFun, primalsFlat, tangentsFlat);
	if (outTree.value === void 0) throw new Error("outTree was not set in jvp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const tangentsOut = unflatten(outTree.value, tangentsOutFlat);
	if (hasAux) return [
		primalsOut,
		tangentsOut,
		lowerAux(aux.value)
	];
	return [primalsOut, tangentsOut];
}
/** Lowering for auxiliary data returned in `hasAux: true` methods. */
function lowerAux(aux) {
	const level = currentTraceLevel();
	return map((x) => {
		if (x instanceof Tracer) while (x._trace.main.level > level) if (x instanceof JVPTracer) {
			x.tangent.dispose();
			x = x.primal;
		} else {
			const y = x.fullLower();
			if (y._trace.main.level >= x._trace.main.level) throw new Error("internal: lowerAux did not reduce trace level");
			x = y;
		}
		return x;
	}, aux);
}

//#endregion
//#region src/frontend/linearize.ts
/** Array value that can either be known or unknown. */
var PartialVal = class PartialVal {
	constructor(val, aval) {
		this.val = val;
		this.aval = aval;
	}
	static known(val) {
		return new PartialVal(val, ShapedArray.fromAval(val.aval));
	}
	static unknown(aval) {
		return new PartialVal(null, ShapedArray.fromAval(aval));
	}
	get isKnown() {
		return this.val !== null;
	}
	toString() {
		return this.val ? this.val.toString() : this.aval.toString();
	}
};
function partialEvalFlat(f, pvalsIn) {
	const main = newMain(PartialEvalTrace);
	const trace$1 = new PartialEvalTrace(main);
	const tracersIn = pvalsIn.map((pval) => trace$1.newArg(pval));
	const unknownTracersIn = tracersIn.filter((t) => !t.pval.isKnown).map((t) => t.ref);
	const outs = f(...tracersIn);
	const tracersOut = outs.map((out) => fullRaise(trace$1, out));
	const pvalsOut = tracersOut.map((t) => t.pval);
	const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
	const jaxpr = partialEvalGraphToJaxpr(unknownTracersIn, unknownTracersOut);
	return {
		jaxpr,
		pvalsOut
	};
}
/**
* Helper function with shared Jaxpr logic between linearize and vjp.
*
* Internally, vjp() looks very similar to linearize() but returns a function
* evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
* of pushing forward tangents.
*/
function linearizeFlatUtil(f, primalsIn) {
	const pvalsIn = [...primalsIn.map(PartialVal.known), ...primalsIn.map((t) => PartialVal.unknown(t.aval))];
	const fJvp = (...x) => {
		const k = x.length / 2;
		const [primalsOut$1, tangentsOut] = jvp$1(f, x.slice(0, k), x.slice(k, 2 * k));
		return [...primalsOut$1, ...tangentsOut];
	};
	const { jaxpr, pvalsOut } = partialEvalFlat(fJvp, pvalsIn);
	const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
	if (!primalPvals.every((pval) => pval.isKnown)) throw new Error("Not all primal values are known after partial evaluation");
	const primalsOut = primalPvals.map((pval) => pval.val);
	return {
		primalsOut,
		jaxpr
	};
}
function linearizeFlat(f, primalsIn) {
	const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
	const fLin = (...tangents) => evalJaxpr(jaxpr.jaxpr, [...jaxpr.consts.map((c) => c.ref), ...tangents]);
	const dispose$1 = () => jaxpr.dispose();
	return [
		primalsOut,
		fLin,
		dispose$1
	];
}
function linearize$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, fLinFlat, dispose$1] = linearizeFlat(fFlat, primalsInFlat.map(pureArray));
	if (outTree.value === void 0) throw new Error("outTree was not set in linearize");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fLin = ((...tangentsIn) => {
		const [tangentsInFlat, inTree2] = flatten(tangentsIn);
		if (!inTree.equals(inTree2)) throw new TreeMismatchError("linearize", inTree, inTree2);
		const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
		return unflatten(outTree.value, tangentsOutFlat);
	});
	fLin.dispose = dispose$1;
	if (hasAux) return [
		primalsOut,
		fLin,
		lowerAux(aux.value)
	];
	return [primalsOut, fLin];
}
var PartialEvalTracer = class extends Tracer {
	#rc;
	constructor(trace$1, pval, recipe) {
		super(trace$1);
		this.pval = pval;
		this.recipe = recipe;
		this.#rc = 1;
	}
	get aval() {
		return this.pval.aval;
	}
	toString() {
		if (!this.recipe) return `PartialEvalTracer(${this.pval.toString()})`;
		else return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
	}
	get ref() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc++;
		return this;
	}
	dispose() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		if (--this.#rc === 0) {
			if (this.pval.isKnown) this.pval.val.dispose();
			else if (this.recipe) {
				if (this.recipe.type === "Const") this.recipe.val.dispose();
				else if (this.recipe.type === "JaxprEqn") this.recipe.tracersIn.forEach((t) => t.dispose());
			}
		}
	}
	fullLower() {
		if (this.pval.isKnown) {
			const val = this.pval.val.ref;
			this.dispose();
			return val;
		}
		return this;
	}
};
var PartialEvalTrace = class extends Trace {
	newArg(pval) {
		if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
		return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
	}
	pure(val) {
		return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
	}
	lift = this.pure;
	instantiateConst(tracer) {
		if (!tracer.pval.isKnown) return tracer;
		else {
			const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
			const val = tracer.pval.val.ref;
			tracer.dispose();
			return new PartialEvalTracer(this, pval, {
				type: "Const",
				val
			});
		}
	}
	processPrimitive(primitive, tracers, params) {
		if (tracers.every((t) => t.pval.isKnown)) return bind(primitive, tracers.map((t) => t.fullLower()), params);
		if (primitive === Primitive.Jit) {
			const { name, jaxpr, numConsts } = params;
			return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
		}
		if (primitive === Primitive.Scan) return this.#partialEvalScan(params, tracers);
		const tracersIn = tracers.map((t) => this.instantiateConst(t));
		const avalsIn = tracersIn.map((t) => t.pval.aval);
		const avalsOut = abstractEvalRules[primitive](avalsIn, params);
		const recipe = {
			type: "JaxprEqn",
			prim: primitive,
			tracersIn,
			params,
			avalsOut,
			tracerRefsOut: []
		};
		const tracersOut = avalsOut.map((aval, i) => {
			if (i > 0) tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
		});
		recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
		return tracersOut;
	}
	/**
	* Partial eval for Scan primitive.
	*
	* When scan is encountered during partial evaluation (e.g., inside JVP for VJP):
	* - If all inputs are known, just run the scan
	* - If this is a JVP'd scan (doubled carry/xs), we can split primal (known)
	*   from tangent (unknown) outputs
	* - Otherwise, mark all outputs as unknown
	*/
	#partialEvalScan(params, tracers) {
		const { jaxpr: _jaxpr, numConsts, numCarry, length: _length, reverse: _reverse } = params;
		const isKnown = tracers.map((t) => t.pval.isKnown);
		const hasUnknown = isKnown.some((k) => !k);
		if (!hasUnknown) {
			const inputs = tracers.map((t) => t.fullLower());
			return bind(Primitive.Scan, inputs, params);
		}
		const avalsIn = tracers.map((t) => t.pval.aval);
		const avalsOut = abstractEvalRules[Primitive.Scan](avalsIn, params);
		const numY = avalsOut.length - numCarry;
		const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0;
		if (!isJvpScan) {
			const tracersIn$1 = tracers.map((t) => this.instantiateConst(t));
			const recipe$1 = {
				type: "JaxprEqn",
				prim: Primitive.Scan,
				tracersIn: tracersIn$1,
				params,
				avalsOut,
				tracerRefsOut: []
			};
			const tracersOut$1 = avalsOut.map((aval, i) => {
				if (i > 0) tracersIn$1.forEach((t) => t.ref);
				return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe$1);
			});
			recipe$1.tracerRefsOut = tracersOut$1.map((t) => new WeakRef(t));
			return tracersOut$1;
		}
		const numPrimalCarry = numCarry / 2;
		const numPrimalY = numY / 2;
		const numX = tracers.length - numConsts - numCarry;
		numX / 2;
		const fullInputs = tracers.map((t) => {
			if (t.pval.isKnown) return t.pval.val.ref;
			else return zeros(t.pval.aval.shape, { dtype: t.pval.aval.dtype });
		});
		const fullOuts = bind(Primitive.Scan, fullInputs, params);
		const tracersIn = tracers.map((t) => this.instantiateConst(t));
		const recipe = {
			type: "JaxprEqn",
			prim: Primitive.Scan,
			tracersIn,
			params,
			avalsOut,
			tracerRefsOut: []
		};
		const tracersOut = [];
		for (let i = 0; i < numPrimalCarry; i++) tracersOut.push(new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null));
		let isFirstUnknown = true;
		for (let i = numPrimalCarry; i < numCarry; i++) {
			fullOuts[i].dispose();
			if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
			isFirstUnknown = false;
			tracersOut.push(new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe));
		}
		for (let i = 0; i < numPrimalY; i++) tracersOut.push(new PartialEvalTracer(this, PartialVal.known(fullOuts[numCarry + i]), null));
		for (let i = numPrimalY; i < numY; i++) {
			fullOuts[numCarry + i].dispose();
			tracersIn.forEach((t) => t.ref);
			tracersOut.push(new PartialEvalTracer(this, PartialVal.unknown(avalsOut[numCarry + i]), recipe));
		}
		recipe.tracerRefsOut = tracersOut.map((t) => t.pval.isKnown ? null : new WeakRef(t));
		return tracersOut;
	}
	/**
	* Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
	* values as possible (with JIT) and forwarding the unknown ones.
	*
	* Used when encountering a Jit rule during the trace.
	*/
	#partialEvalJaxpr(name, jaxpr, numConsts, tracers) {
		jaxpr = jaxpr.flatten();
		const inUnknowns = tracers.map((t) => !t.pval.isKnown);
		const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(jaxpr, inUnknowns);
		const [knownTracers, unknownTracers] = require_backend.partitionList(inUnknowns, tracers);
		const outs1Res = bind(Primitive.Jit, knownTracers.map((t) => t.ref.fullLower()), {
			name: `${name}_peval`,
			jaxpr: jaxpr1,
			numConsts: 0
		});
		const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
		const res = outs1Res.slice(jaxpr1.outs.length - numRes);
		const resTracers = res.map((x) => this.instantiateConst(fullRaise(this, x)));
		const recipe = {
			type: "JaxprEqn",
			prim: Primitive.Jit,
			tracersIn: resTracers.concat(unknownTracers),
			params: {
				name: `${name}_resid`,
				jaxpr: jaxpr2,
				numConsts: 0
			},
			avalsOut: jaxpr2.outs.map((x) => x.aval),
			tracerRefsOut: []
		};
		const outs2 = jaxpr2.outs.map((x, i$1) => {
			if (i$1 > 0) recipe.tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
		});
		recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));
		let i = 0;
		let j = 0;
		return outUnknowns.map((unk) => unk ? outs2[j++] : outs1[i++]);
	}
};
/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(jaxpr, inUnknowns, instantiate) {
	jaxpr = jaxpr.flatten();
	const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
	const knownVars = new Set(knownIns);
	const residuals = /* @__PURE__ */ new Set();
	const eqns1 = [];
	const eqns2 = [];
	for (const eqn of jaxpr.eqns) {
		if (eqn.primitive === Primitive.Jit) throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
		const hasUnknowns = eqn.inputs.some((x) => x instanceof Var && !knownVars.has(x));
		if (hasUnknowns) {
			for (const x of eqn.inputs) if (x instanceof Var && knownVars.has(x)) residuals.add(x);
			eqns2.push(eqn);
		} else {
			eqns1.push(eqn);
			for (const v of eqn.outBinders) knownVars.add(v);
		}
	}
	const outUnknowns = jaxpr.outs.map((x) => x instanceof Var && !knownVars.has(x));
	if (instantiate !== void 0) for (let i = 0; i < jaxpr.outs.length; i++) {
		const x = jaxpr.outs[i];
		if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
			residuals.add(x);
			outUnknowns[i] = true;
		}
	}
	const residualsL = Array.from(residuals);
	const [ins1, ins2] = require_backend.partitionList(inUnknowns, jaxpr.inBinders);
	const [outs1, outs2] = require_backend.partitionList(outUnknowns, jaxpr.outs);
	const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
	const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
	return {
		jaxpr1,
		jaxpr2,
		outUnknowns,
		numRes: residualsL.length
	};
}
/**
* Convert the graph representation of a partial eval to a standard Jaxpr.
* Also called `tracers_to_jaxpr()` in JAX.
*/
function partialEvalGraphToJaxpr(tracersIn, tracersOut) {
	const tracerToVar = /* @__PURE__ */ new Map();
	const constToVar = /* @__PURE__ */ new Map();
	const processedEqns = /* @__PURE__ */ new Set();
	const eqns = [];
	for (const t of tracersIn) tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
	for (const t of require_backend.toposort(tracersOut, (t$1) => t$1.recipe?.type === "JaxprEqn" ? t$1.recipe.tracersIn : [])) {
		if (!t.recipe) throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
		if (t.recipe.type === "LambdaBinding") {
			if (!tracersIn.includes(t)) throw new TypeError("LambdaBinding tracer not in input list");
		} else if (t.recipe.type === "Const") {
			const val = t.recipe.val;
			let binder = constToVar.get(val);
			if (!binder) {
				binder = new Var(ShapedArray.fromAval(val.aval));
				constToVar.set(val, binder);
			}
			tracerToVar.set(t, binder);
		} else if (t.recipe.type === "JaxprEqn") {
			if (!processedEqns.has(t.recipe)) {
				processedEqns.add(t.recipe);
				const tracersIn$1 = t.recipe.tracersIn.map((t$1) => tracerToVar.get(t$1));
				const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
				for (let i = 0; i < outBinders.length; i++) {
					const ref$1 = t.recipe.tracerRefsOut[i];
					const tracerOut = ref$1?.deref?.();
					if (tracerOut) tracerToVar.set(tracerOut, outBinders[i]);
				}
				eqns.push(new JaxprEqn(t.recipe.prim, tracersIn$1, t.recipe.params, outBinders));
			}
		}
	}
	const [consts, constvars] = require_backend.unzip2(constToVar.entries());
	const inBinders = [...constvars, ...tracersIn.map((t) => tracerToVar.get(t))];
	const outVars = tracersOut.map((t) => tracerToVar.get(t));
	let jaxpr = new Jaxpr(inBinders, eqns, outVars);
	typecheckJaxpr(jaxpr);
	for (const t of consts) t.ref;
	for (const t of tracersIn) t.dispose();
	for (const t of tracersOut) t.dispose();
	jaxpr = jaxpr.simplify();
	if (require_backend.DEBUG >= 5) console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
	return new ClosedJaxpr(jaxpr, consts);
}
/** Marker type for pullback, used by transpose rules. */
var UndefPrimal = class {
	aval;
	constructor(aval) {
		this.aval = ShapedArray.fromAval(aval);
	}
};
/**
* Helper to get or compute a primal (known) variable's value during transpose.
* For intermediate variables that are known (computed from only known inputs),
* we need to evaluate the equations that produce them.
*/
function getOrComputePrimal(jaxpr, knownVars, knownPrimals, v) {
	if (knownPrimals.has(v)) return knownPrimals.get(v).ref;
	const eqn = jaxpr.eqns.find((eq) => eq.outBinders.some((out) => out === v));
	if (!eqn) throw new Error(`Internal error: could not find equation producing variable`);
	const inputVals = eqn.inputs.map((inp) => inp instanceof Lit ? array(inp.value, { dtype: inp.dtype }) : getOrComputePrimal(jaxpr, knownVars, knownPrimals, inp));
	const results = bind(eqn.primitive, inputVals, eqn.params);
	for (let i = 0; i < eqn.outBinders.length; i++) knownPrimals.set(eqn.outBinders[i], results[i]);
	const result = knownPrimals.get(v);
	if (!result) throw new Error(`Internal error: variable not produced by equation`);
	return result.ref;
}
/**
* Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
*
* Will raise a TypeError if the provided Jaxpr is not a linear function of its,
* inputs, as general expressions cannot be transposed.
*/
function evalJaxprTransposed(jaxpr, args, cotangents) {
	const knownVars = /* @__PURE__ */ new Set();
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (!(args[i] instanceof UndefPrimal)) knownVars.add(jaxpr.inBinders[i]);
	for (const eqn of jaxpr.eqns) {
		const allInputsKnown = eqn.inputs.every((v) => v instanceof Lit || knownVars.has(v));
		if (allInputsKnown) for (const outVar of eqn.outBinders) knownVars.add(outVar);
	}
	const knownPrimals = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (!(args[i] instanceof UndefPrimal)) knownPrimals.set(jaxpr.inBinders[i], args[i]);
	const ctStore = /* @__PURE__ */ new Map();
	const readCotangent = (v) => {
		const ct = ctStore.get(v);
		if (ct) {
			ctStore.delete(v);
			return ct;
		} else return zeros(v.aval.shape, { dtype: v.aval.dtype });
	};
	const writeCotangent = (v, ct) => {
		if (ct !== null) {
			const oldCt = ctStore.get(v);
			if (oldCt) ctStore.set(v, add$1(oldCt, ct));
			else ctStore.set(v, ct);
		}
	};
	for (let i = 0; i < jaxpr.outs.length; i++) {
		const v = jaxpr.outs[i];
		if (v instanceof Var) writeCotangent(v, cotangents[i]);
	}
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		const allInputsKnown = eqn.inputs.every((v) => v instanceof Lit || knownVars.has(v));
		if (allInputsKnown) continue;
		const primalsIn = eqn.inputs.map((v) => v instanceof Lit ? array(v.value, { dtype: v.dtype }) : knownVars.has(v) ? getOrComputePrimal(jaxpr, knownVars, knownPrimals, v) : new UndefPrimal(v.aval));
		const cotangentsOut = eqn.outBinders.map(readCotangent);
		const rule = transposeRules[eqn.primitive];
		if (!rule) throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
		const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
		for (let j = 0; j < eqn.inputs.length; j++) {
			const v = eqn.inputs[j];
			if (v instanceof Var && !knownVars.has(v)) writeCotangent(v, cotangentsIn[j]);
			else if (cotangentsIn[j] !== null) throw new Error("internal: cotangent should be null");
		}
	}
	for (const t of knownPrimals.values()) t.dispose();
	const results = [];
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (args[i] instanceof UndefPrimal) results.push(readCotangent(jaxpr.inBinders[i]));
	return results;
}
/**
* Inverse operation of `generalBroadcast()` for backpropagation.
*
* `x` has the shape of the result of an operation that was broadcasted with
* `target` (it's a cotangent during backprop). Returns a tracer with rank and
* shape equal to `target`.
*/
function unbroadcast(x, target) {
	const shape$1 = target.aval.shape;
	const extraDims = x.ndim > shape$1.length ? require_backend.range(x.ndim - shape$1.length) : [];
	if (x.ndim < shape$1.length) throw new Error(`unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape$1})`);
	const unsqueeze = [];
	const keptReduceDims = [];
	for (let i = 0; i < shape$1.length; i++) {
		const indexFromEnd = shape$1.length - i;
		const indexInX = x.ndim - indexFromEnd;
		const xLen = x.shape[indexInX];
		if (xLen > 1 && shape$1[i] === 1) {
			unsqueeze.push(i);
			keptReduceDims.push(indexInX);
		} else if (shape$1[i] !== xLen) throw new Error("internal: unbroadcast shape mismatch");
	}
	const reductionDims = [...extraDims, ...keptReduceDims];
	if (reductionDims.length === 0) return x;
	let result = x.sum(reductionDims);
	if (!require_backend.deepEqual(result.shape, shape$1)) result = broadcast(result, shape$1, unsqueeze);
	return result;
}
var NonlinearError = class extends TypeError {
	constructor(primitive) {
		super(`Nonlinear operation in backward pass for ${primitive}`);
	}
};
const transposeRules = {
	[Primitive.Mul]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError(Primitive.Mul);
		return [x instanceof UndefPrimal ? unbroadcast(mul(ct, y), x) : null, y instanceof UndefPrimal ? unbroadcast(mul(x, ct), y) : null];
	},
	[Primitive.Neg]([ct], [x]) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
		return [neg(ct)];
	},
	[Primitive.Add]([ct], [x, y]) {
		if (!(x instanceof UndefPrimal || y instanceof UndefPrimal)) throw new NonlinearError(Primitive.Add);
		if (x instanceof UndefPrimal && y instanceof UndefPrimal) return [unbroadcast(ct.ref, x), unbroadcast(ct, y)];
		return x instanceof UndefPrimal ? (y.dispose(), [unbroadcast(ct, x), null]) : (x.dispose(), [null, unbroadcast(ct, y)]);
	},
	[Primitive.Reduce]([ct], [x], { op, axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reduce);
		if (op === require_backend.AluOp.Add) return [broadcast(ct, x.aval.shape, axis)];
		else throw new NonlinearError(Primitive.Reduce);
	},
	[Primitive.Pool]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pool);
		return bind(Primitive.PoolTranspose, [ct], {
			inShape: x.aval.shape,
			window,
			strides
		});
	},
	[Primitive.PoolTranspose]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.PoolTranspose);
		return bind(Primitive.Pool, [ct], {
			window,
			strides
		});
	},
	[Primitive.Dot]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError(Primitive.Dot);
		const axisSize = require_backend.generalBroadcast(x.aval.shape, y.aval.shape).slice(-1)[0];
		ct = broadcast(ct, ct.shape.concat(axisSize), [-1]);
		return [x instanceof UndefPrimal ? unbroadcast(mul(ct, y), x) : null, y instanceof UndefPrimal ? unbroadcast(mul(x, ct), y) : null];
	},
	[Primitive.Conv]([ct], [lhs, rhs], params) {
		if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal) throw new NonlinearError(Primitive.Conv);
		const v = params.vmapDims;
		const rev01 = [
			...require_backend.range(v),
			v + 1,
			v,
			...require_backend.range(v + 2, ct.ndim)
		];
		if (lhs instanceof UndefPrimal) {
			let kernel = rhs;
			kernel = transpose$1(kernel, rev01);
			kernel = flip$1(kernel, require_backend.range(v + 2, kernel.ndim));
			const result = conv$1(ct, kernel, {
				vmapDims: v,
				strides: params.lhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedKernel = (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
					const padBefore = dilatedKernel - 1 - pl;
					const dilatedLhs = (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
					const padAfter = dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
					return [padBefore, padAfter];
				}),
				lhsDilation: params.strides,
				rhsDilation: params.rhsDilation
			});
			return [result, null];
		} else {
			const newLhs = transpose$1(lhs, rev01);
			const newRhs = transpose$1(ct, rev01);
			let result = conv$1(newLhs, newRhs, {
				vmapDims: v,
				strides: params.rhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedLhs = (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
					const dilatedKernel = (rhs.aval.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
					const padFromLhs = dilatedCt - dilatedLhs;
					const padFromRhs = dilatedKernel - pl - 1;
					return [pl, padFromLhs + padFromRhs];
				}),
				lhsDilation: params.lhsDilation,
				rhsDilation: params.strides
			});
			result = transpose$1(result, rev01);
			return [null, result];
		}
	},
	[Primitive.Where]([ct], [cond, x, y]) {
		const cts = [
			null,
			null,
			null
		];
		if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
		if (x instanceof UndefPrimal) cts[1] = unbroadcast(where$1(cond.ref, ct.ref, 0), x);
		else x.dispose();
		if (y instanceof UndefPrimal) cts[2] = unbroadcast(where$1(cond.ref, 0, ct.ref), y);
		else y.dispose();
		ct.dispose();
		cond.dispose();
		return cts;
	},
	[Primitive.Concatenate]([ct], inputs, { axis }) {
		if (inputs.some((x) => !(x instanceof UndefPrimal))) throw new NonlinearError(Primitive.Concatenate);
		const sizes = inputs.map((x) => x.aval.shape[axis]);
		return split$2(ct, axis, sizes);
	},
	[Primitive.Split](cts, [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Split);
		return [concatenate$1(cts, axis)];
	},
	[Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
		if (indices.some((i) => i instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
		throw new Error("Gather transpose rule is not yet implemented, requires complex Scatter sum operation");
	},
	[Primitive.Transpose]([ct], [x], { perm }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Transpose);
		return [transpose$1(ct, require_backend.invertPermutation(perm))];
	},
	[Primitive.Broadcast]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Broadcast);
		return [reduce(ct, require_backend.AluOp.Add, axis)];
	},
	[Primitive.Reshape]([ct], [x], _) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reshape);
		return [reshape$1(ct, x.aval.shape)];
	},
	[Primitive.Flip]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
		return [flip$1(ct, axis)];
	},
	[Primitive.Shrink]([ct], [x], { slice }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Shrink);
		const width = slice.map(([s, e$1], i) => [s, x.aval.shape[i] - e$1]);
		return [pad$1(ct, width)];
	},
	[Primitive.Pad]([ct], [x], { width }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pad);
		const slice = width.map(([s, _e], i) => [s, s + x.aval.shape[i]]);
		return [shrink(ct, slice)];
	},
	[Primitive.TriangularSolve]([ct], [a, b], { unitDiagonal }) {
		if (a instanceof UndefPrimal || !(b instanceof UndefPrimal)) throw new NonlinearError(Primitive.TriangularSolve);
		const ctB = triangularSolve$1(moveaxis(a, -2, -1), ct, {
			lower: true,
			unitDiagonal
		});
		return [null, ctB];
	},
	[Primitive.Jit](cts, args, { name, jaxpr }) {
		const undefPrimals = args.map((x) => x instanceof UndefPrimal);
		const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
		const residuals = args.filter((x, i$1) => !undefPrimals[i$1]);
		const outs = bind(Primitive.Jit, [
			...newJaxpr.consts.map((c) => c.ref),
			...residuals,
			...cts
		], {
			name: `${name}_t`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		let i = 0;
		return undefPrimals.map((isUndef) => isUndef ? outs[i++] : null);
	},
	[Primitive.Scan](cts, args, { jaxpr, numCarry, numConsts, length, reverse }) {
		const numX = args.length - numConsts - numCarry;
		const numY = cts.length - numCarry;
		const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0 && numX % 2 === 0;
		const numPrimalCarry = isJvpScan ? numCarry / 2 : 0;
		const numPrimalX = isJvpScan ? numX / 2 : 0;
		const numPrimalY = isJvpScan ? numY / 2 : 0;
		const undefMask = args.map((x, i) => {
			if (x instanceof UndefPrimal) return true;
			if (!isJvpScan) return false;
			if (i < numConsts) return false;
			else if (i < numConsts + numCarry) {
				const carryIdx = i - numConsts;
				return carryIdx >= numPrimalCarry;
			} else {
				const xIdx = i - numConsts - numCarry;
				return xIdx >= numPrimalX;
			}
		});
		const bodyNumConsts = numConsts;
		const bodyNumCarry = numCarry;
		const bodyUndefPrimals = [];
		for (let i = 0; i < jaxpr.inBinders.length; i++) if (i < bodyNumConsts) bodyUndefPrimals.push(undefMask[i]);
		else if (i < bodyNumConsts + bodyNumCarry) bodyUndefPrimals.push(undefMask[numConsts + (i - bodyNumConsts)]);
		else bodyUndefPrimals.push(undefMask[numConsts + numCarry + (i - bodyNumConsts - bodyNumCarry)]);
		const constArgs = args.slice(0, numConsts);
		const carryArgs = args.slice(numConsts, numConsts + numCarry);
		const xsArgs = args.slice(numConsts + numCarry);
		const constResiduals = constArgs.filter((_, i) => !undefMask[i]);
		const carryResiduals = carryArgs.filter((_, i) => !undefMask[numConsts + i]);
		const xsResiduals = xsArgs.filter((_, i) => !undefMask[numConsts + numCarry + i]);
		const carryIsPrimal = carryArgs.map((_, i) => !undefMask[numConsts + i]);
		const xsIsPrimal = xsArgs.map((_, i) => !undefMask[numConsts + numCarry + i]);
		const actualNumPrimalCarry = isJvpScan ? numPrimalCarry : carryIsPrimal.filter((x) => x).length;
		isJvpScan ? numPrimalX : xsIsPrimal.filter((x) => x).length;
		if (actualNumPrimalCarry === 0 || carryResiduals.length === 0) throw new Error("Scan transpose: no carry residuals available. grad() through scan requires primal carry values to be available as residuals.");
		constResiduals.length;
		const halfOutputs = jaxpr.outs.length / 2;
		Math.floor(halfOutputs);
		const forwardInTypes = jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).map((v) => v.aval);
		const { jaxpr: primalForwardJaxpr } = makeJaxpr$1((...primalInputs) => {
			const fullInputs = [];
			let primalIdx = 0;
			for (let i = 0; i < jaxpr.inBinders.length; i++) if (bodyUndefPrimals[i]) {
				const aval = jaxpr.inBinders[i].aval;
				fullInputs.push(zeros(aval.shape, { dtype: aval.dtype }));
			} else fullInputs.push(primalInputs[primalIdx++].ref);
			const outs = evalJaxpr(jaxpr, fullInputs);
			const primalCarryOuts = outs.slice(0, numPrimalCarry);
			const primalYOuts = outs.slice(numCarry, numCarry + Math.floor(numY / 2));
			for (let i = numPrimalCarry; i < numCarry; i++) outs[i].dispose();
			for (let i = numCarry + Math.floor(numY / 2); i < outs.length; i++) outs[i].dispose();
			return [...primalCarryOuts, ...primalYOuts];
		})(...forwardInTypes);
		const allCarries = [];
		let currentCarry = carryResiduals.map((c) => c.ref);
		allCarries.push(currentCarry.map((c) => c.ref));
		for (let iter = 0; iter < length; iter++) {
			const dataIdx = reverse ? length - 1 - iter : iter;
			const xSlices = [];
			for (const xs of xsResiduals) {
				const slice = shrink(xs.ref, [[dataIdx, dataIdx + 1], ...xs.shape.slice(1).map((_, i) => [0, xs.shape[i + 1]])]);
				xSlices.push(reshape$1(slice, xs.shape.slice(1)));
			}
			const forwardInputs = [
				...constResiduals.map((c) => c.ref),
				...currentCarry.map((c) => c.ref),
				...xSlices
			];
			const forwardOuts = evalJaxpr(primalForwardJaxpr.jaxpr, [...primalForwardJaxpr.consts.map((c) => c.ref), ...forwardInputs]);
			const newCarry = forwardOuts.slice(0, numPrimalCarry);
			for (let i = numPrimalCarry; i < forwardOuts.length; i++) forwardOuts[i].dispose();
			for (const c of currentCarry) c.dispose();
			currentCarry = newCarry;
			allCarries.push(currentCarry.map((c) => c.ref));
		}
		for (const c of currentCarry) c.dispose();
		const numTangentConsts = numConsts - constResiduals.length;
		const numTangentCarry = numCarry - numPrimalCarry;
		const numTangentX = numX - numPrimalX;
		numY - numPrimalY;
		const tangentBodyInAvals = [...jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).map((v) => v.aval), ...jaxpr.inBinders.filter((_, i) => bodyUndefPrimals[i]).map((v) => v.aval)];
		const { jaxpr: tangentBody } = makeJaxpr$1((...tangentBodyArgs) => {
			const numPrimalInputs = jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).length;
			const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
			const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);
			const fullInputs = [];
			let primalIdx = 0;
			let tangentIdx = 0;
			for (let i = 0; i < jaxpr.inBinders.length; i++) if (bodyUndefPrimals[i]) fullInputs.push(tangentInputs[tangentIdx++].ref);
			else fullInputs.push(primalResiduals[primalIdx++].ref);
			const fullOuts = evalJaxpr(jaxpr, fullInputs);
			const tangentOuts = [];
			for (let i = numPrimalCarry; i < numCarry; i++) tangentOuts.push(fullOuts[i]);
			for (let i = numCarry + numPrimalY; i < fullOuts.length; i++) tangentOuts.push(fullOuts[i]);
			for (let i = 0; i < numPrimalCarry; i++) fullOuts[i].dispose();
			for (let i = numCarry; i < numCarry + numPrimalY; i++) fullOuts[i].dispose();
			return tangentOuts;
		})(...tangentBodyInAvals);
		const tangentBodyUndefPrimals = [...Array(tangentBody.jaxpr.inBinders.length - (numTangentConsts + numTangentCarry + numTangentX)).fill(false), ...Array(numTangentConsts + numTangentCarry + numTangentX).fill(true)];
		const transposedBody = transposeJaxpr(tangentBody.jaxpr, tangentBodyUndefPrimals);
		const ctCarryAll = cts.slice(0, numCarry);
		const ctYsAll = cts.slice(numCarry);
		let ctCarryRunning = ctCarryAll.slice(numPrimalCarry).map((c) => c.ref);
		for (let i = 0; i < numPrimalCarry; i++) ctCarryAll[i].dispose();
		const ctXsAccum = [];
		for (let i = 0; i < numTangentX; i++) ctXsAccum.push([]);
		let ctConstsAccum = null;
		for (let iter = length - 1; iter >= 0; iter--) {
			const dataIdx = reverse ? length - 1 - iter : iter;
			const primalCarry = allCarries[iter];
			const xSlices = [];
			for (const xs of xsResiduals) {
				const slice = shrink(xs.ref, [[dataIdx, dataIdx + 1], ...xs.shape.slice(1).map((_, i) => [0, xs.shape[i + 1]])]);
				xSlices.push(reshape$1(slice, xs.shape.slice(1)));
			}
			const ctYSlices = [];
			ctYsAll.length - Math.floor(numY / 2);
			for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) {
				const ctY = ctYsAll[i];
				const slice = shrink(ctY.ref, [[dataIdx, dataIdx + 1], ...ctY.shape.slice(1).map((_, j) => [0, ctY.shape[j + 1]])]);
				ctYSlices.push(reshape$1(slice, ctY.shape.slice(1)));
			}
			const bodyOutCotangents = [];
			bodyOutCotangents.push(...ctCarryRunning.map((c) => c.ref));
			bodyOutCotangents.push(...ctYSlices);
			const transposedInputs = [
				...transposedBody.consts.map((c) => c.ref),
				...constResiduals.map((c) => c.ref),
				...primalCarry.map((c) => c.ref),
				...xSlices,
				...bodyOutCotangents
			];
			const transposedOuts = evalJaxpr(transposedBody.jaxpr, transposedInputs);
			let outIdx = 0;
			const ctConstsIter = [];
			for (let i = 0; i < numTangentConsts; i++) ctConstsIter.push(transposedOuts[outIdx++]);
			const ctCarryNew = [];
			const numTangentCarry$1 = numCarry - numPrimalCarry;
			for (let i = 0; i < numTangentCarry$1; i++) ctCarryNew.push(transposedOuts[outIdx++]);
			const ctXIter = [];
			for (let i = 0; i < numTangentX; i++) ctXIter.push(transposedOuts[outIdx++]);
			if (ctConstsAccum === null) ctConstsAccum = ctConstsIter;
			else ctConstsAccum = ctConstsAccum.map((ct, i) => add$1(ct, ctConstsIter[i]));
			for (let i = 0; i < numTangentX; i++) ctXsAccum[i].push(ctXIter[i]);
			for (const c of ctCarryRunning) c.dispose();
			ctCarryRunning = ctCarryNew;
			for (const c of primalCarry) c.dispose();
		}
		for (const c of allCarries[length]) c.dispose();
		for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) ctYsAll[i].dispose();
		for (let i = 0; i < Math.floor(numY / 2); i++) ctYsAll[i].dispose();
		const ctXsStacked = [];
		for (let i = 0; i < numTangentX; i++) {
			const reversed = ctXsAccum[i].reverse();
			if (reverse) reversed.reverse();
			const expanded = reversed.map((ct) => broadcast(ct, [1, ...ct.shape], [0]));
			const stacked = concatenate$1(expanded, 0);
			ctXsStacked.push(stacked);
		}
		const actualUndefMask = args.map((x) => x instanceof UndefPrimal);
		const result = [];
		let ctConstIdx = 0;
		let ctCarryIdx = 0;
		let ctXIdx = 0;
		for (let i = 0; i < args.length; i++) {
			const isJvpTangent = undefMask[i];
			if (!actualUndefMask[i]) {
				if (isJvpTangent) if (i < numConsts) ctConstsAccum[ctConstIdx++].dispose();
				else if (i < numConsts + numCarry) ctCarryRunning[ctCarryIdx++].dispose();
				else ctXsStacked[ctXIdx++].dispose();
				result.push(null);
			} else if (i < numConsts) result.push(ctConstsAccum[ctConstIdx++]);
			else if (i < numConsts + numCarry) result.push(ctCarryRunning[ctCarryIdx++]);
			else result.push(ctXsStacked[ctXIdx++]);
		}
		primalForwardJaxpr.dispose();
		transposedBody.dispose();
		for (const c of constResiduals) c.dispose();
		for (const c of carryResiduals) c.dispose();
		for (const c of xsResiduals) c.dispose();
		return result;
	}
};
const transposeJaxprCache = /* @__PURE__ */ new Map();
function transposeJaxpr(jaxpr, undefPrimals) {
	const cacheKey = JSON.stringify(undefPrimals);
	const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
	const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
	const { jaxpr: newJaxpr } = makeJaxpr$1((forwardIn, cotangents) => {
		const args = [];
		let forwardInIdx = 0;
		for (let i = 0; i < undefPrimals.length; i++) if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
		else args.push(forwardIn[forwardInIdx++]);
		return evalJaxprTransposed(jaxpr, args, cotangents);
	})(forwardInTypes, outTypes);
	typecheckJaxpr(newJaxpr.jaxpr);
	if (!transposeJaxprCache.has(jaxpr)) transposeJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	transposeJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vjpFlat(f, primalsIn) {
	const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
	const fVjp = (...cotangents) => {
		const transposeInputs = [...jaxpr.consts.map((c) => c.ref), ...primalsIn.map((t) => new UndefPrimal(t.aval))];
		return evalJaxprTransposed(jaxpr.jaxpr, transposeInputs, cotangents);
	};
	const dispose$1 = () => jaxpr.dispose();
	return [
		primalsOut,
		fVjp,
		dispose$1
	];
}
function vjp$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, fVjpFlat, dispose$1] = vjpFlat(fFlat, primalsInFlat.map(pureArray));
	if (outTree.value === void 0) throw new Error("outTree was not set in vjp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fVjp = ((cotangentsOut) => {
		const [cotangentsOutFlat, outTree2] = flatten(cotangentsOut);
		if (!outTree.value.equals(outTree2)) throw new TreeMismatchError("vjp", outTree.value, outTree2);
		const cotangentsInFlat = fVjpFlat(...cotangentsOutFlat.map(pureArray));
		return unflatten(inTree, cotangentsInFlat);
	});
	fVjp.dispose = dispose$1;
	if (hasAux) return [
		primalsOut,
		fVjp,
		lowerAux(aux.value)
	];
	return [primalsOut, fVjp];
}
function grad$1(f, opts) {
	const valueAndGradFn = valueAndGrad$1(f, opts);
	return (...x) => {
		if (opts?.hasAux) {
			const [[y, aux], dx] = valueAndGradFn(...x);
			y.dispose();
			return [dx, aux];
		} else {
			const [y, dx] = valueAndGradFn(...x);
			y.dispose();
			return dx;
		}
	};
}
function valueAndGrad$1(f, opts) {
	const argnums = opts?.argnums ?? 0;
	const hasAux = opts?.hasAux ?? false;
	require_backend.checkInts(argnums);
	const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
	return (...x) => {
		if (x.length === 0) throw new Error("grad requires at least one argument to differentiate");
		for (let i = 0; i < x.length; i++) if (!argnumsSet.has(i)) x[i] = map(stopGradient, x[i]);
		const [y, fVjp, aux] = vjp$1(f, x, { hasAux });
		if (!(y instanceof Tracer) || ndim$1(y) !== 0) throw new TypeError("grad requires a scalar output");
		if (!require_backend.isFloatDtype(y.dtype)) throw new TypeError("grad only supports floating-point dtypes");
		const cts = fVjp(onesLike$1(y.ref));
		fVjp.dispose();
		for (let i = 0; i < cts.length; i++) if (!argnumsSet.has(i)) dispose(cts[i]);
		const grads = typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
		return hasAux ? [[y, aux], grads] : [y, grads];
	};
}
function jacrev$1(f) {
	return function jacobianReverse(x) {
		if (x.shape.length !== 1) throw new TypeError("jacrev only supports 1D inputs");
		const [size$1] = x.shape;
		const pullback = (ct) => {
			const [y, fVjp] = vjp$1(f, [x]);
			y.dispose();
			const [ret] = fVjp(ct);
			fVjp.dispose();
			return ret;
		};
		return vmap$1(pullback, [1])(eye(size$1, void 0, { dtype: x.dtype }));
	};
}
function hessian$1(f) {
	return jacfwd$1(grad$1(f));
}

//#endregion
//#region src/library/numpy/einsum.ts
const bprod = (...xs) => xs.reduce((acc, x) => acc * BigInt(x), 1n);
const uniq = (arr) => Array.from(new Set(arr));
const EINSUM_COMPONENT_RE = /\p{ID_Start}|\.\.\./gu;
const einsumParseCache = /* @__PURE__ */ new Map();
/** Parse an Einstein notation string into a runnable `EinsumInput`. */
function parseEinsumExpression(expr, shapes) {
	return require_backend.runWithCache(einsumParseCache, [expr, shapes], () => {
		const idents = [...expr.split("->")[0].matchAll(EINSUM_COMPONENT_RE).map((m) => m[0]).filter((c) => c !== "...")];
		if (!expr.includes("->")) {
			const counts = /* @__PURE__ */ new Map();
			for (const c of idents) counts.set(c, (counts.get(c) ?? 0) + 1);
			const outputIndices = Array.from(counts.entries()).filter(([, count]) => count === 1).map(([char]) => char).sort();
			if (expr.includes("...")) outputIndices.splice(0, 0, "...");
			expr += "->" + outputIndices.join("");
		}
		const identToIndex = new Map(uniq(idents).sort().map((c, i) => [c, i]));
		const componentsToIndices = (components, rank) => components.flatMap((c) => {
			if (c === "...") {
				const start = rank !== void 0 ? components.length - 1 + ellipsisRank - rank : 0;
				return require_backend.range(identToIndex.size + start, identToIndex.size + ellipsisRank);
			}
			return identToIndex.get(c);
		});
		let ellipsisRank = 0;
		const [lhs, rhs] = expr.split("->");
		const lhsComponents = lhs.split(",").map((part) => [...part.matchAll(EINSUM_COMPONENT_RE).map((m) => m[0])]);
		const rhsComponents = [...rhs.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]);
		for (const [i, components] of lhsComponents.entries()) {
			const shape$1 = shapes[i];
			const ellipsisIndex = components.indexOf("...");
			if (ellipsisIndex !== -1) {
				if (components.lastIndexOf("...") !== ellipsisIndex) throw new Error("Multiple ellipses in one einsum operand is not allowed");
				const numExplicit = components.length - 1;
				if (shape$1.length < numExplicit) throw new Error(`Einsum operand ${i} has shape ${JSON.stringify(shape$1)} but indexed with "${components.join("")}"`);
				ellipsisRank = Math.max(ellipsisRank, shape$1.length - numExplicit);
			}
		}
		const lhsIndices = lhsComponents.map((components, i) => componentsToIndices(components, shapes[i].length));
		const rhsIndex = componentsToIndices(rhsComponents);
		return {
			shapes,
			lhsIndices,
			rhsIndex
		};
	});
}
var EinsumPath = class {
	/** Parsed and normalized input for the einsum. */
	input;
	/** Mapping of each index number to its size in the shape array. */
	sizeMap;
	/**
	* A list of tensor contractions.
	*
	* This is ordered by operation order. Each entry corresponds to a single
	* elementwise product and/or inner contraction between two tensors, and it
	* contains the indices of the tensors to be contracted.
	*
	* The indices of input tensors are [0..n), and each intermediate from the
	* path at index i produces a new tensor at index n + i at the end
	* (opt_einsum internally calls this "SSA form").
	*
	* Invariants:
	* - Each group in the path consists of two tensors.
	* - For n input tensors, there are n-1 groups in the path.
	* - Every tensor must be in the path exactly once, except the final output.
	*
	* @example
	* Given einsum for `(A, B, C)`, this path corresponds to `(A, B)` and then
	* `(AB, C)`.
	* ```
	* [[0, 1], [3, 2]]
	* ```
	*/
	path;
	constructor(input, sizeMap, path) {
		this.input = input;
		this.sizeMap = sizeMap;
		this.path = path;
	}
	/** Shape of the final output tensor. */
	get outputShape() {
		return this.input.rhsIndex.map((i) => this.sizeMap.get(i));
	}
	/** Estimate the number of FLOPs to execute this einsum path. */
	get approximateFlops() {
		return approximatePathFlops(this.input, this.sizeMap, this.path);
	}
};
function approximatePathFlops(input, sizeMap, path) {
	if (path.length == 0) {
		const [indices$1] = input.lhsIndices;
		return bprod(...uniq(indices$1).map((i) => sizeMap.get(i)));
	}
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	let totalFlops = 0n;
	for (const tensorGroup of path) {
		const indexReduced = [];
		const indexGroup = [];
		for (const tensorIdx of tensorGroup) for (const idx of indices[tensorIdx]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		totalFlops += approximateCountFlops(indexGroup, indexReduced.length > 0, tensorGroup.length, sizeMap);
		const newIndex = indexGroup.filter((x) => !indexReduced.includes(x));
		for (const idx of newIndex) indexUsageCounts[idx]++;
		indices.push(newIndex);
	}
	return totalFlops;
}
function approximateCountFlops(indexGroup, hasReduction, numTerms, sizeMap) {
	const elements = bprod(...indexGroup.map((i) => sizeMap.get(i)));
	const flopsPerLoopIteration = BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n);
	return elements * flopsPerLoopIteration;
}
/** Compute size for each index in the einsum expression, also validates input. */
function computeSizeMap({ shapes, lhsIndices, rhsIndex }) {
	if (shapes.length === 0) throw new Error("Einsum must have at least one input tensor");
	if (lhsIndices.length !== shapes.length) throw new Error(`Mismatched number of lhs operands (${lhsIndices.length}) and shapes (${shapes.length})`);
	for (let i = 0; i < shapes.length; i++) if (lhsIndices[i].length !== shapes[i].length) throw new Error(`Mismatched number of indices (${lhsIndices[i].length}) and shape (${JSON.stringify(shapes[i])}) for operand ${i}`);
	const rhsIndexSet = /* @__PURE__ */ new Set();
	for (const idx of rhsIndex) {
		if (rhsIndexSet.has(idx)) throw new Error(`Repeated index ${idx} in einsum output`);
		rhsIndexSet.add(idx);
	}
	const sizeMap = /* @__PURE__ */ new Map();
	for (let i = 0; i < shapes.length; i++) {
		const shape$1 = shapes[i];
		const lhsIndex = lhsIndices[i];
		for (let j = 0; j < lhsIndex.length; j++) {
			const idx = lhsIndex[j];
			const dim = shape$1[j];
			const existing = sizeMap.get(idx);
			if (existing === void 0 || existing === 1) sizeMap.set(idx, dim);
			else if (existing !== dim && dim !== 1) throw new Error(`Inconsistent size for index ${idx} in einsum: ${existing} vs ${dim}`);
		}
	}
	for (const [idx, size$1] of sizeMap) if (!Number.isInteger(idx) || idx < 0) throw new Error(`Invalid index ${idx} in einsum expression, must be non-negative integer`);
	else if (size$1 < 0) throw new Error(`Invalid size ${size$1} for index ${idx} in einsum expression, must be non-negative`);
	for (const idx of rhsIndex) if (!sizeMap.has(idx)) throw new Error(`Output index ${idx} not present in einsum inputs`);
	return sizeMap;
}
const einsumPathCache = /* @__PURE__ */ new Map();
function computeEinsumPath(input, method) {
	if (!method) method = input.shapes.length <= 5 ? "optimal" : "naive";
	return require_backend.runWithCache(einsumPathCache, [input, method], () => {
		const sizeMap = computeSizeMap(input);
		if (input.shapes.length === 1) return new EinsumPath(input, sizeMap, []);
		switch (method) {
			case "naive": return computePathNaive(input, sizeMap);
			case "optimal": return computePathOptimal(input, sizeMap);
			default: throw new Error(`Unknown computePath method: ${method}`);
		}
	});
}
function computePathNaive(input, sizeMap) {
	const n = input.shapes.length;
	const path = [];
	let lastTensorIndex = 0;
	for (let i = 1; i < n; i++) {
		path.push([lastTensorIndex, i]);
		lastTensorIndex = n + i - 1;
	}
	return new EinsumPath(input, sizeMap, path);
}
function computePathOptimal(input, sizeMap) {
	const n = input.shapes.length;
	let bestPath = null;
	let bestFlops = null;
	for (const path of allPaths(require_backend.range(n), n)) {
		const flops = approximatePathFlops(input, sizeMap, path);
		if (bestFlops === null || flops < bestFlops) {
			bestPath = path;
			bestFlops = flops;
		}
	}
	return new EinsumPath(input, sizeMap, bestPath);
}
function* allPaths(tensors, next) {
	if (tensors.length === 2) {
		yield [[tensors[0], tensors[1]]];
		return;
	}
	for (let i = 0; i < tensors.length; i++) for (let j = i + 1; j < tensors.length; j++) {
		const pair = [tensors[i], tensors[j]];
		const newTensors = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
		newTensors.push(next);
		for (const subpath of allPaths(newTensors, next + 1)) yield [pair, ...subpath];
	}
}

//#endregion
//#region src/library/numpy-fft.ts
var numpy_fft_exports = {};
__export(numpy_fft_exports, {
	fft: () => fft,
	ifft: () => ifft
});
function checkPairInput(name, a) {
	const fullName = `jax.numpy.fft.${name}`;
	if (!require_backend.deepEqual(a.real.shape, a.imag.shape)) throw new Error(`${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`);
	if (a.real.dtype !== a.imag.dtype) throw new Error(`${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`);
	if (!require_backend.isFloatDtype(a.real.dtype)) throw new Error(`${fullName}: input must have a float dtype, got ${a.real.dtype}`);
}
function checkPowerOfTwo(name, n) {
	if ((n & n - 1) !== 0) throw new Error(`jax.numpy.fft.${name}: size must be a power of two, got ${n}`);
}
const fftUpdate = jit$1(function fftUpdate$1(i, { real, imag }) {
	const half = 2 ** i;
	real = real.reshape([-1, 2 * half]);
	imag = imag.reshape([-1, 2 * half]);
	const k = arange(0, half, 1, { dtype: real.dtype });
	const theta = k.mul(-Math.PI / half);
	const wr = cos(theta.ref);
	const wi = sin(theta);
	const ur = real.ref.slice([], [0, half]);
	const ui = imag.ref.slice([], [0, half]);
	const vr = real.slice([], [half, 2 * half]);
	const vi = imag.slice([], [half, 2 * half]);
	const tr = vr.ref.mul(wr.ref).sub(vi.ref.mul(wi.ref));
	const ti = vr.mul(wi).add(vi.mul(wr));
	return {
		real: concatenate([ur.ref.add(tr.ref), ur.sub(tr)], -1),
		imag: concatenate([ui.ref.add(ti.ref), ui.sub(ti)], -1)
	};
}, { staticArgnums: [0] });
/**
* Compute a one-dimensional discrete Fourier transform.
*
* Currently, the size of the axis must be a power of two.
*/
function fft(a, axis = -1) {
	checkPairInput("fft", a);
	let { real, imag } = a;
	axis = require_backend.checkAxis(axis, real.ndim);
	const n = real.shape[axis];
	checkPowerOfTwo("fft", n);
	const logN = Math.log2(n);
	let perm = null;
	if (axis !== real.ndim - 1) {
		perm = require_backend.range(real.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		real = real.transpose(perm);
		imag = imag.transpose(perm);
	}
	const originalShape = real.shape;
	real = real.reshape([-1, ...require_backend.rep(logN, 2)]).transpose([0, ...require_backend.range(1, logN + 1).reverse()]).flatten();
	imag = imag.reshape([-1, ...require_backend.rep(logN, 2)]).transpose([0, ...require_backend.range(1, logN + 1).reverse()]).flatten();
	for (let i = 0; i < logN; i++) ({real, imag} = fftUpdate(i, {
		real,
		imag
	}));
	real = real.reshape(originalShape);
	imag = imag.reshape(originalShape);
	if (perm !== null) {
		real = real.transpose(require_backend.invertPermutation(perm));
		imag = imag.transpose(require_backend.invertPermutation(perm));
	}
	return {
		real,
		imag
	};
}
/**
* Compute a one-dimensional inverse discrete Fourier transform.
*
* Currently, the size of the axis must be a power of two.
*/
function ifft(a, axis = -1) {
	checkPairInput("ifft", a);
	let { real, imag } = a;
	axis = require_backend.checkAxis(axis, real.ndim);
	const n = real.shape[axis];
	checkPowerOfTwo("ifft", n);
	imag = imag.mul(-1);
	const result = fft({
		real,
		imag
	}, axis);
	return {
		real: result.real.div(n),
		imag: result.imag.mul(-1).div(n)
	};
}

//#endregion
//#region src/library/numpy-linalg.ts
var numpy_linalg_exports = {};
__export(numpy_linalg_exports, {
	cholesky: () => cholesky,
	det: () => det,
	diagonal: () => diagonal,
	inv: () => inv,
	lstsq: () => lstsq,
	matmul: () => matmul,
	matrixPower: () => matrixPower,
	matrixTranspose: () => matrixTranspose,
	outer: () => outer,
	slogdet: () => slogdet,
	solve: () => solve,
	tensordot: () => tensordot,
	trace: () => trace,
	vecdot: () => vecdot
});
function checkSquare(name, a) {
	if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) throw new Error(`${name}: input must be at least 2D square matrix, got ${a.aval}`);
	return a.shape[a.ndim - 1];
}
/**
* Compute the Cholesky decomposition of a (batched) positive-definite matrix.
*
* This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
* the input matrix, which is on by default.
*/
function cholesky(a, { upper = false, symmetrizeInput = true } = {}) {
	a = fudgeArray(a);
	checkSquare("cholesky", a);
	if (symmetrizeInput) a = a.ref.add(matrixTranspose(a)).mul(.5);
	return cholesky$1(a, { upper });
}
/** Compute the determinant of a square matrix (batched). */
function det(a) {
	a = fudgeArray(a);
	const n = checkSquare("det", a);
	const [lu$2, pivots, permutation] = lu(a);
	permutation.dispose();
	const parity = pivots.notEqual(arange(n)).astype(int32).sum(-1).mod(2);
	const sign$1 = parity.mul(-2).add(1);
	const diag$1 = lu$2.diagonal(0, -1, -2);
	return prod$1(diag$1, -1).mul(sign$1);
}
/** Compute the inverse of a square matrix (batched). */
function inv(a) {
	a = fudgeArray(a);
	const n = checkSquare("inv", a);
	return solve(a, eye(n));
}
/**
* Return the least-squares solution to a linear equation.
*
* For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
* For underdetermined systems, this finds the minimum-norm solution for `x`.
*
* This currently uses Cholesky decomposition to solve the normal equations,
* under the hood. The method is not as robust as QR or SVD.
*
* @param a coefficient matrix of shape `(M, N)`
* @param b right-hand side of shape `(M,)` or `(M, K)`
* @return least-squares solution of shape `(N,)` or `(N, K)`
*/
function lstsq(a, b) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	if (a.ndim !== 2) throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
	const [m, n] = a.shape;
	if (b.shape[0] !== m) throw new Error(`lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`);
	const at = matrixTranspose(a.ref);
	if (m <= n) {
		const aat = matmul(a, at.ref);
		const l = cholesky(aat, { symmetrizeInput: false });
		const lb = triangularSolve(l.ref, b, {
			leftSide: true,
			lower: true
		});
		const llb = triangularSolve(l, lb, {
			leftSide: true,
			transposeA: true
		});
		return matmul(at, llb.ref);
	} else {
		const ata = matmul(at.ref, a);
		const l = cholesky(ata, { symmetrizeInput: false });
		const atb = matmul(at, b);
		const lb = triangularSolve(l.ref, atb, {
			leftSide: true,
			lower: true
		});
		const llb = triangularSolve(l, lb, {
			leftSide: true,
			transposeA: true
		});
		return llb;
	}
}
/** Raise a square matrix to an integer power, via repeated squarings. */
function matrixPower(a, n) {
	if (!Number.isInteger(n)) throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
	a = fudgeArray(a);
	const m = checkSquare("matrixPower", a);
	if (n === 0) {
		a.dispose();
		return broadcastTo(eye(m), a.shape);
	}
	if (n < 0) {
		a = inv(a);
		n = -n;
	}
	let result = null;
	let a2k = a;
	for (let k = 0; n; k++) {
		if (k > 0) a2k = matmul(a2k.ref, a2k);
		if (n % 2 === 1) result = result === null ? a2k.ref : matmul(result, a2k.ref);
		n = Math.floor(n / 2);
	}
	a2k.dispose();
	return result;
}
/** Return sign and natural logarithm of the determinant of `a`. */
function slogdet(a) {
	a = fudgeArray(a);
	const n = checkSquare("slogdet", a);
	const [lu$2, pivots, permutation] = lu(a);
	permutation.dispose();
	let parity = pivots.notEqual(arange(n)).astype(int32).sum(-1);
	const diag$1 = lu$2.diagonal(0, -1, -2);
	parity = parity.add(diag$1.ref.less(0).astype(int32).sum(-1)).mod(2);
	const logabsdet = log(absolute(diag$1)).sum(-1);
	const sign$1 = parity.mul(-2).add(1);
	return [sign$1, logabsdet];
}
/**
* Solve a linear system of equations.
*
* This solves a (batched) linear system of equations `a @ x = b` for `x` given
* `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
*
* @param a - Coefficient matrix of shape `(..., N, N)`.
* @param b - Values of shape `(N,)` or `(..., N, M)`.
* @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
*/
function solve(a, b) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	const n = checkSquare("solve", a);
	if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
	const bIs1d = b.ndim === 1;
	if (bIs1d) b = b.reshape([...b.shape, 1]);
	if (b.shape[b.ndim - 2] !== n) throw new Error(`solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`);
	const m = b.shape[b.ndim - 1];
	const batchDims = require_backend.generalBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2));
	a = broadcastTo(a, [
		...batchDims,
		n,
		n
	]);
	b = broadcastTo(b, [
		...batchDims,
		n,
		m
	]);
	const [lu$2, pivots, permutation] = lu(a);
	pivots.dispose();
	const P = arange(n).equal(permutation.reshape([...permutation.shape, 1])).astype(b.dtype);
	const LPb = triangularSolve(lu$2.ref, matmul(P, b), {
		leftSide: true,
		lower: true,
		unitDiagonal: true
	});
	let x = triangularSolve(lu$2, LPb.ref, {
		leftSide: true,
		lower: false
	});
	if (bIs1d) x = squeeze(x, -1);
	return x;
}

//#endregion
//#region src/library/numpy/dtype-info.ts
/** Machine limits for floating-point types. */
function finfo(dtype) {
	if (!require_backend.isFloatDtype(dtype)) throw new Error(`finfo: received ${dtype}, must be a floating-point type`);
	switch (dtype) {
		case require_backend.DType.Float16: return Object.freeze({
			bits: 16,
			dtype: require_backend.DType.Float16,
			eps: 2 ** -10,
			epsneg: 2 ** -11,
			machep: -10,
			max: 65504,
			maxexp: 16,
			min: -65504,
			minexp: -14,
			negep: -24,
			nexp: 5,
			nmant: 10,
			precision: 3,
			resolution: .001,
			smallestNormal: 2 ** -14,
			smallestSubnormal: 2 ** -24
		});
		case require_backend.DType.Float32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Float32,
			eps: 2 ** -23,
			epsneg: 2 ** -24,
			machep: -23,
			max: 34028234663852886e22,
			maxexp: 128,
			min: -34028234663852886e22,
			minexp: -126,
			negep: -24,
			nexp: 8,
			nmant: 23,
			precision: 6,
			resolution: 1e-6,
			smallestNormal: 2 ** -126,
			smallestSubnormal: 2 ** -149
		});
		case require_backend.DType.Float64: return Object.freeze({
			bits: 64,
			dtype: require_backend.DType.Float64,
			eps: 2 ** -52,
			epsneg: 2 ** -53,
			machep: -52,
			max: Number.MAX_VALUE,
			maxexp: 1024,
			min: -Number.MAX_VALUE,
			minexp: -1022,
			negep: -53,
			nexp: 11,
			nmant: 52,
			precision: 15,
			resolution: 1e-15,
			smallestNormal: 2 ** -1022,
			smallestSubnormal: 2 ** -1074
		});
		default: throw new Error(`finfo: unsupported dtype ${dtype}`);
	}
}
/** Machine limits for integer types. */
function iinfo(dtype) {
	switch (dtype) {
		case require_backend.DType.Int32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Int32,
			max: 2147483647,
			min: -2147483648
		});
		case require_backend.DType.Uint32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Uint32,
			max: 4294967295,
			min: 0
		});
		default: throw new Error(`iinfo: unsupported dtype ${dtype}`);
	}
}

//#endregion
//#region src/library/numpy.ts
var numpy_exports = {};
__export(numpy_exports, {
	Array: () => Array$1,
	DType: () => require_backend.DType,
	abs: () => absolute,
	absolute: () => absolute,
	acos: () => acos,
	acosh: () => arccosh,
	add: () => add,
	all: () => all,
	allclose: () => allclose,
	any: () => any,
	arange: () => arange,
	arccos: () => acos,
	arccosh: () => arccosh,
	arcsin: () => asin,
	arcsinh: () => arcsinh,
	arctan: () => atan,
	arctan2: () => atan2,
	arctanh: () => arctanh,
	argmax: () => argmax,
	argmin: () => argmin,
	argsort: () => argsort,
	array: () => array,
	asin: () => asin,
	asinh: () => arcsinh,
	astype: () => astype,
	atan: () => atan,
	atan2: () => atan2,
	atanh: () => arctanh,
	bool: () => bool,
	broadcastArrays: () => broadcastArrays,
	broadcastShapes: () => broadcastShapes,
	broadcastTo: () => broadcastTo,
	cbrt: () => cbrt,
	ceil: () => ceil,
	clip: () => clip,
	columnStack: () => columnStack,
	concatenate: () => concatenate,
	convolve: () => convolve,
	corrcoef: () => corrcoef,
	correlate: () => correlate,
	cos: () => cos,
	cosh: () => cosh,
	cov: () => cov,
	cumsum: () => cumsum,
	cumulativeSum: () => cumsum,
	deg2rad: () => deg2rad,
	degrees: () => degrees,
	diag: () => diag,
	diagonal: () => diagonal,
	divide: () => trueDivide,
	divmod: () => divmod,
	dot: () => dot$1,
	dstack: () => dstack,
	e: () => e,
	einsum: () => einsum,
	equal: () => equal,
	eulerGamma: () => eulerGamma,
	exp: () => exp,
	exp2: () => exp2,
	expandDims: () => expandDims,
	expm1: () => expm1,
	eye: () => eye,
	fft: () => numpy_fft_exports,
	finfo: () => finfo,
	flip: () => flip,
	fliplr: () => fliplr,
	flipud: () => flipud,
	float16: () => float16,
	float32: () => float32,
	float64: () => float64,
	floor: () => floor,
	floorDivide: () => floorDivide,
	fmod: () => fmod,
	frexp: () => frexp,
	full: () => full,
	fullLike: () => fullLike$1,
	greater: () => greater,
	greaterEqual: () => greaterEqual,
	hamming: () => hamming,
	hann: () => hann,
	heaviside: () => heaviside,
	hstack: () => hstack,
	hypot: () => hypot,
	identity: () => identity$1,
	iinfo: () => iinfo,
	inf: () => inf,
	inner: () => inner,
	int32: () => int32,
	isfinite: () => isfinite,
	isinf: () => isinf,
	isnan: () => isnan,
	isneginf: () => isneginf,
	isposinf: () => isposinf,
	ldexp: () => ldexp,
	less: () => less,
	lessEqual: () => lessEqual,
	linalg: () => numpy_linalg_exports,
	linspace: () => linspace,
	log: () => log,
	log10: () => log10,
	log1p: () => log1p,
	log2: () => log2,
	logspace: () => logspace,
	matmul: () => matmul,
	matrixTranspose: () => matrixTranspose,
	max: () => max,
	maximum: () => maximum,
	mean: () => mean,
	meshgrid: () => meshgrid,
	min: () => min,
	minimum: () => minimum,
	moveaxis: () => moveaxis$1,
	multiply: () => multiply,
	nan: () => nan,
	nanToNum: () => nanToNum,
	ndim: () => ndim,
	negative: () => negative,
	notEqual: () => notEqual,
	ones: () => ones,
	onesLike: () => onesLike,
	outer: () => outer,
	pad: () => pad,
	permuteDims: () => transpose,
	pi: () => pi,
	positive: () => positive,
	pow: () => power,
	power: () => power,
	prod: () => prod$1,
	promoteTypes: () => require_backend.promoteTypes,
	ptp: () => ptp,
	rad2deg: () => rad2deg,
	radians: () => radians,
	ravel: () => ravel,
	reciprocal: () => reciprocal,
	remainder: () => remainder,
	repeat: () => repeat,
	reshape: () => reshape,
	shape: () => shape,
	sign: () => sign,
	sin: () => sin,
	sinc: () => sinc,
	sinh: () => sinh,
	size: () => size,
	sort: () => sort,
	split: () => split$1,
	sqrt: () => sqrt,
	square: () => square,
	squeeze: () => squeeze,
	stack: () => stack,
	std: () => std,
	subtract: () => subtract,
	sum: () => sum,
	swapaxes: () => swapaxes,
	take: () => take,
	tan: () => tan,
	tanh: () => tanh,
	tensordot: () => tensordot,
	tile: () => tile,
	trace: () => trace,
	transpose: () => transpose,
	tri: () => tri,
	tril: () => tril,
	triu: () => triu,
	trueDivide: () => trueDivide,
	trunc: () => trunc,
	uint32: () => uint32,
	var_: () => var_,
	vdot: () => vdot,
	vecdot: () => vecdot,
	vstack: () => vstack,
	where: () => where,
	zeros: () => zeros,
	zerosLike: () => zerosLike
});
const float32 = require_backend.DType.Float32;
const int32 = require_backend.DType.Int32;
const uint32 = require_backend.DType.Uint32;
const bool = require_backend.DType.Bool;
const float16 = require_backend.DType.Float16;
const float64 = require_backend.DType.Float64;
/** Euler's constant, `e = 2.7182818284590...` */
const e = Math.E;
/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
const eulerGamma = .5772156649015329;
/** Positive infinity. */
const inf = Number.POSITIVE_INFINITY;
/** Floating-point representation of NaN. */
const nan = NaN;
/** This is Pi, `π = 3.14159265358979...` */
const pi = Math.PI;
/** @function Element-wise addition, with broadcasting. */
const add = add$1;
/** @function Element-wise multiplication, with broadcasting. */
const multiply = mul;
/** @function Numerical negative of every element of an array. */
const negative = neg;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
const reciprocal = reciprocal$1;
/** @function Round input down to the nearest integer. */
const floor = floor$1;
/** @function Round input up to the nearest integer. */
const ceil = ceil$1;
/** @function Element-wise sine function (takes radians). */
const sin = sin$1;
/** @function Element-wise cosine function (takes radians). */
const cos = cos$1;
/** @function Element-wise inverse sine function (inverse of sin). */
const asin = asin$1;
/** @function Element-wise inverse tangent function (inverse of tan). */
const atan = atan$1;
/** @function Calculate the exponential of all elements in the input array. */
const exp = exp$1;
/** @function Calculate the natural logarithm of all elements in the input array. */
const log = log$1;
/** @function Calculate the square root of all elements in the input array. */
const sqrt = sqrt$1;
/** @function Return element-wise minimum of the input arrays. */
const minimum = min$1;
/** @function Return element-wise maximum of the input arrays. */
const maximum = max$1;
/** @function Compare two arrays element-wise. */
const greater = greater$1;
/** @function Compare two arrays element-wise. */
const less = less$1;
/** @function Compare two arrays element-wise. */
const equal = equal$1;
/** @function Compare two arrays element-wise. */
const notEqual = notEqual$1;
/** @function Compare two arrays element-wise. */
const greaterEqual = greaterEqual$1;
/** @function Compare two arrays element-wise. */
const lessEqual = lessEqual$1;
/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
const where = where$1;
/**
* @function
* Permute the dimensions of an array. Defaults to reversing the axis order.
*/
const transpose = transpose$1;
/**
* @function
* Give a new shape to an array without changing its data.
*
* One shape dimension can be -1. In this case, the value is inferred from the
* length of the array and remaining dimensions.
*/
const reshape = reshape$1;
/**
* @function
* Move axes of an array to new positions. Other axes retain original order.
*/
const moveaxis$1 = moveaxis;
/**
* @function
* Add padding (zeros) to an array.
*
* The `width` argument is either an integer or pair of integers, in which case
* all axes are padded with the same width. Or if it is an array of pairs, each
* pair specifies the padding for its corresponding axis.
*/
const pad = pad$1;
/**
* @function
* Return the number of dimensions of an array. Does not consume array reference.
*/
const ndim = ndim$1;
/** @function Return the shape of an array. Does not consume array reference. */
const shape = getShape;
/**
* @function
* Return an array of zeros with the same shape and type as a given array.
*/
const zerosLike = zerosLike$1;
/**
* @function
* Return an array of ones with the same shape and type as a given array.
*/
const onesLike = onesLike$1;
/**
* @function
* Return a full array with the same shape and type as a given array.
*/
const fullLike$1 = fullLike;
/**
* Return the number of elements in an array, optionally along an axis.
* Does not consume array reference.
*/
function size(a, axis) {
	const s = shape(a);
	return axis === void 0 ? require_backend.prod(s) : s[axis];
}
/** Convert an array to a specified dtype. */
function astype(a, dtype) {
	return fudgeArray(a).astype(dtype);
}
/** Sum of the elements of the array over a given axis, or axes. */
function sum(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Add, axis, opts);
}
/** Product of the array elements over a given axis. */
function prod$1(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Mul, axis, opts);
}
/** Return the minimum of array elements along a given axis. */
function min(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Min, axis, opts);
}
/** Return the maximum of array elements along a given axis. */
function max(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Max, axis, opts);
}
/**
* Test whether any array element along a given axis evaluates to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function any(a, axis = null, opts) {
	return fudgeArray(a).any(axis, opts);
}
/**
* Test whether all array elements along a given axis evaluate to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function all(a, axis = null, opts) {
	return fudgeArray(a).all(axis, opts);
}
/** Return the peak-to-peak range along a given axis (`max - min`). */
function ptp(a, axis = null, opts) {
	a = fudgeArray(a);
	return max(a.ref, axis, opts).sub(min(a, axis, opts));
}
/** Compute the average of the array elements along the specified axis. */
function mean(a, axis = null, opts) {
	return fudgeArray(a).mean(axis, opts);
}
/**
* Returns the indices of the minimum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmin(a, axis, opts) {
	a = fudgeArray(a);
	if (axis === void 0) {
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const shape$1 = a.shape;
	const isMax = equal(a, min(a.ref, axis, { keepdims: true }));
	const length = array(shape$1[axis], {
		dtype: int32,
		device: a.device
	});
	const idx = isMax.astype(require_backend.DType.Int32).mul(arange(shape$1[axis], 0, -1, {
		dtype: int32,
		device: a.device
	}).reshape([shape$1[axis], ...require_backend.rep(shape$1.length - axis - 1, 1)]));
	return length.sub(max(idx, axis, opts));
}
/**
* Returns the indices of the maximum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmax(a, axis, opts) {
	a = fudgeArray(a);
	if (axis === void 0) {
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const shape$1 = a.shape;
	const isMax = equal(a, max(a.ref, axis, { keepdims: true }));
	const length = array(shape$1[axis], {
		dtype: int32,
		device: a.device
	});
	const idx = isMax.astype(require_backend.DType.Int32).mul(arange(shape$1[axis], 0, -1, {
		dtype: int32,
		device: a.device
	}).reshape([shape$1[axis], ...require_backend.rep(shape$1.length - axis - 1, 1)]));
	return length.sub(max(idx, axis, opts));
}
/**
* Cumulative sum of elements along an axis.
*
* Currently this function is `O(n^2)`, we'll improve this later on with a
* two-phase parallel reduction algorithm.
*/
function cumsum(a, axis) {
	a = fudgeArray(a);
	if (axis === void 0) {
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const n = a.shape[axis];
	a = moveaxis$1(a, axis, -1);
	a = broadcast(a, a.shape.concat(n), [-2]);
	return moveaxis$1(tril(a).sum(-1), -1, axis);
}
/** Reverse the elements in an array along the given axes. */
function flip(x, axis = null) {
	const nd = ndim(x);
	axis = require_backend.normalizeAxis(axis, nd);
	return flip$1(x, axis);
}
/**
* Split an array into multiple sub-arrays along an axis.
*
* @param a - The input array to split.
* @param indicesOrSections - If an integer, it indicates the number of equal
* sections to create along the specified axis. If a list of integers, it
* specifies the indices at which to split the array.
* @param axis - The axis along which to split the array. Default is 0.
*/
function split$1(a, indicesOrSections, axis = 0) {
	a = fudgeArray(a);
	axis = require_backend.checkAxis(axis, a.ndim);
	const size$1 = a.shape[axis];
	let sizes;
	if (typeof indicesOrSections === "number") {
		if (size$1 % indicesOrSections !== 0) throw new Error(`Array of size ${size$1} cannot be split into ${indicesOrSections} equal parts`);
		const partSize = size$1 / indicesOrSections;
		sizes = require_backend.rep(indicesOrSections, partSize);
	} else {
		const indices = indicesOrSections.map((i) => i < 0 ? i + size$1 : i);
		sizes = [indices[0]];
		for (let i = 1; i < indices.length; i++) sizes.push(indices[i] - indices[i - 1]);
		sizes.push(size$1 - indices[indices.length - 1]);
	}
	const results = [];
	for (let i = 0; i < sizes.length; i += 7) if (i === sizes.length) {
		results.push(a);
		break;
	} else if (i + 8 >= sizes.length) {
		results.push(...split$2(a, axis, sizes.slice(i)));
		break;
	} else {
		const groupSizes = [...sizes.slice(i, i + 7), sizes.slice(i + 7).reduce((x, y) => x + y, 0)];
		const outs = split$2(a, axis, groupSizes);
		results.push(...outs.slice(0, -1));
		a = outs[outs.length - 1];
	}
	return results;
}
/**
* Join a sequence of arrays along an existing axis.
*
* The arrays must have the same shape, except in the dimension corresponding to
* `axis` (the first, by default).
*
* No scalars can be passed to this function, as the axis is then ambiguous.
*/
function concatenate(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to concatenate");
	const shapes = xs.map(shape);
	axis = require_backend.checkAxis(axis, shapes[0].length);
	for (let i = 1; i < shapes.length; i++) if (shapes[i].length !== shapes[0].length || !shapes[i].every((d, j) => j === axis || d === shapes[0][j])) throw new Error(`Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`);
	let result = xs[0];
	for (let i = 1; i < xs.length; i += 7) {
		const group = xs.slice(i, i + 7);
		result = concatenate$1([result, ...group], axis);
	}
	return result;
}
/**
* Join a sequence of arrays along a new axis.
*
* The `axis` parameter specifies the index of the new axis in the dimensions of
* the result. For example, if `axis=0` it will be the first dimension and if
* `axis=-1` it will be the last dimension.
*
* All shapes must have the same shape.
*/
function stack(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to stack");
	const shapes = xs.map((x) => shape(x));
	if (!shapes.every((s) => require_backend.deepEqual(s, shapes[0]))) throw new Error(`Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`);
	axis = require_backend.checkAxis(axis, shapes[0].length + 1);
	const newShape = shapes[0].toSpliced(axis, 0, 1);
	const newArrays = xs.map((x) => fudgeArray(x).reshape(newShape));
	return concatenate(newArrays, axis);
}
/**
* Horizontally stack arrays. Inputs are promoted to rank at least 1, then
* concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
*/
function hstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to hstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs);
	else if (nds[0] === 1) return concatenate(xs);
	else return concatenate(xs, 1);
}
/**
* Vertically stack arrays. Inputs are promoted to rank at least 2, then
* concatenated along axis 0.
*/
function vstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to vstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([-1, 1]);
	else if (nds[0] === 1) return stack(xs);
	else return concatenate(xs);
}
/**
* Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
* concatenated along axis 2.
*/
function dstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to dstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([
		1,
		1,
		-1
	]);
	else if (nds[0] === 1) {
		const ret = stack(xs, -1);
		return ret.reshape([1, ...ret.shape]);
	} else if (nds[0] === 2) return stack(xs, -1);
	else return concatenate(xs, 2);
}
/**
* Stack arrays column-wise. Inputs are promoted to rank at least 2, then
* concatenated along axis 1.
*/
function columnStack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to columnStack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs).reshape([1, -1]);
	else if (nds[0] === 1) return stack(xs, -1);
	else return concatenate(xs, 1);
}
/** Flip an array vertically (axis=0). */
function flipud(x) {
	return flip(x, 0);
}
/** Flip an array horizontally (axis=1). */
function fliplr(x) {
	return flip(x, 1);
}
/** Interchange two axes of an array. */
function swapaxes(a, axis1, axis2) {
	a = fudgeArray(a);
	axis1 = require_backend.checkAxis(axis1, a.ndim);
	axis2 = require_backend.checkAxis(axis2, a.ndim);
	if (axis1 === axis2) return a;
	const perm = require_backend.range(a.ndim);
	perm[axis1] = axis2;
	perm[axis2] = axis1;
	return transpose(a, perm);
}
/** Transpose the last two dimensions of an array. */
function matrixTranspose(a) {
	if (ndim(a) < 2) throw new Error(`matrixTranspose: input array must be at least 2D`);
	return moveaxis$1(a, -1, -2);
}
/** Return a 1-D flattened array containing the elements of the input. */
function ravel(a) {
	return fudgeArray(a).ravel();
}
/** Remove one or more length-1 axes from an array. */
function squeeze(a, axis = null) {
	const as = shape(a);
	if (axis === null) axis = require_backend.range(as.length).filter((i) => as[i] === 1);
	else if (typeof axis === "number") axis = [axis];
	axis = axis.map((a$1) => require_backend.checkAxis(a$1, as.length));
	for (const a$1 of axis) if (as[a$1] !== 1) throw new Error("Cannot squeeze axis with size != 1");
	const newShape = as.filter((_, i) => !axis.includes(i));
	return reshape(a, newShape);
}
/**
* Expand the shape of an array by inserting new axes of length 1.
*
* @param a - Input array.
* @param axis - Position(s) in the expanded axes where the new axis (or axes)
*   is placed. Can be a single integer or an array of integers.
* @returns Array with the number of dimensions increased.
*
* @example
* ```ts
* const x = np.array([1, 2]);
* np.expandDims(x, 0); // Shape [1, 2]
* np.expandDims(x, 1); // Shape [2, 1]
* np.expandDims(x, [0, 2]); // Shape [1, 2, 1]
* ```
*/
function expandDims(a, axis) {
	const as = shape(a);
	axis = typeof axis === "number" ? [axis] : axis;
	axis = require_backend.normalizeAxis(axis, as.length + axis.length);
	const newShape = [];
	let srcIdx = 0;
	for (let i = 0; i < as.length + axis.length; i++) if (axis.includes(i)) newShape.push(1);
	else newShape.push(as[srcIdx++]);
	return reshape(a, newShape);
}
/**
* Repeat each element of an array after themselves.
*
* If no axis is provided, use the flattened input array, and return a flat
* output array.
*/
function repeat(a, repeats, axis) {
	if (!Number.isInteger(repeats) || repeats < 0) throw new Error(`repeat: repeats must be a non-negative integer, got ${repeats}`);
	a = fudgeArray(a);
	if (axis === void 0) {
		a = ravel(a);
		axis = 0;
	}
	axis = require_backend.checkAxis(axis, a.ndim);
	if (repeats === 1) return a;
	const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
	const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
	return broadcast(a, broadcastedShape, [axis + 1]).reshape(finalShape);
}
/**
* Construct an array by repeating A the number of times given by reps.
*
* If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
* integers, the resulting array will have a shape of `(reps[0] * d1,
* reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
*/
function tile(a, reps) {
	a = fudgeArray(a);
	if (typeof reps === "number") reps = [reps];
	if (!reps.every((r) => Number.isInteger(r) && r >= 0)) throw new Error(`tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`);
	const ndiff = reps.length - a.ndim;
	if (ndiff > 0) a = a.reshape([...require_backend.rep(ndiff, 1), ...a.shape]);
	if (ndiff < 0) reps = [...require_backend.rep(-ndiff, 1), ...reps];
	const broadcastedShape = [];
	const broadcastAxes = [];
	for (let i = 0; i < a.ndim; i++) {
		if (reps[i] > 1) {
			broadcastedShape.push(reps[i]);
			broadcastAxes.push(broadcastedShape.length - 1);
		}
		broadcastedShape.push(a.shape[i]);
	}
	const finalShape = a.shape.map((d, i) => reps[i] * d);
	return broadcast(a, broadcastedShape, broadcastAxes).reshape(finalShape);
}
/**
* Broadcast an array to a shape, with NumPy-style broadcasing rules.
*
* In other words, this lets you append axes to the left, and/or expand
* dimensions where the shape is 1.
*/
function broadcastTo(a, shape$1) {
	const nd = ndim(a);
	if (shape$1.length < nd) throw new Error(`broadcastTo: target shape ${JSON.stringify(shape$1)} has fewer dimensions than input array: ${nd}`);
	return broadcast(a, shape$1, require_backend.range(shape$1.length - nd));
}
/** Broadcast input shapes to a common output shape. */
function broadcastShapes(...shapes) {
	if (shapes.length === 0) return [];
	return shapes.reduce(require_backend.generalBroadcast);
}
/** Broadcast arrays to a common shape. */
function broadcastArrays(...arrays) {
	const shapes = arrays.map((a) => shape(a));
	const outShape = broadcastShapes(...shapes);
	return arrays.map((a) => broadcastTo(a, outShape));
}
/**
* Return specified diagonals.
*
* If a is 2D, return the diagonal of the array with the given offset. If a is
* 3D or higher, compute diagonals along the two given axes (default: 0, 1).
*
* This returns a view over the existing array. The shape of the resulting array
* is determined by removing the two axes along which the diagonal is taken,
* then appending a new axis to the right with holding the diagonals.
*/
function diagonal(a, offset, axis1, axis2) {
	return fudgeArray(a).diagonal(offset, axis1, axis2);
}
/**
* Extract a diagonal or construct a diagonal array.
*
* If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
* array, return a 2D array with v on the k-th diagonal.
*/
function diag(v, k = 0) {
	const a = fudgeArray(v);
	if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
	if (a.ndim === 1) {
		const n = a.shape[0];
		const ret = where(eye(n).equal(1), a.ref, zerosLike(a));
		if (k > 0) return pad(ret, [[0, k], [k, 0]]);
		else if (k < 0) return pad(ret, [[-k, 0], [0, -k]]);
		else return ret;
	} else if (a.ndim === 2) return diagonal(a, k);
	else throw new Error("numpy.diag only supports 1D and 2D arrays");
}
/** Calculate the sum of the diagonal of an array along the given axes. */
function trace(a, offset = 0, axis1 = 0, axis2 = 1) {
	return diagonal(a, offset, axis1, axis2).sum(-1);
}
/**
* Return a sorted copy of an array.
*
* The array is sorted along a specified axis (the last by default). This may be
* an unstable sort, and it dispatches to device-specific implementation.
*/
function sort(a, axis = -1) {
	return fudgeArray(a).sort(axis);
}
/**
* Return indices that would sort an array. This may be an unstable sorting
* algorithm; it need not preserve order of indices in ties.
*
* Returns an array of `int32` indices.
*
* The array is sorted along a specified axis (the last by default).
*/
function argsort(a, axis = -1) {
	return fudgeArray(a).argsort(axis);
}
/**
* Take elements from an array along an axis.
*
* This is equivalent to advanced indexing with integer indices over that
* numbered axis. By default, the flattened array is used.
*/
function take(a, indices, axis = null) {
	if (axis === null) {
		a = ravel(a);
		axis = 0;
	}
	axis = require_backend.checkAxis(axis, ndim(a));
	return gather(a, [indices], [axis], axis);
}
/** Return if two arrays are element-wise equal within a tolerance. */
function allclose(actual, expected, options) {
	const { rtol = 1e-5, atol = 1e-7 } = options ?? {};
	const x = array(actual);
	const y = array(expected);
	if (!require_backend.deepEqual(x.shape, y.shape)) return false;
	const xData = x.dataSync();
	const yData = y.dataSync();
	for (let i = 0; i < xData.length; i++) {
		if (isNaN(xData[i]) !== isNaN(yData[i])) return false;
		if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) return false;
	}
	return true;
}
/** Matrix product of two arrays. */
function matmul(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) throw new Error("matmul: x and y must be at least 1D");
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	const numBatchDims = Math.min(Math.max(x.ndim, 2), y.ndim) - 2;
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2],
		lhsBatchDims: require_backend.range(-2 - numBatchDims, -2),
		rhsBatchDims: require_backend.range(-2 - numBatchDims, -2)
	});
}
/** Dot product of two arrays. */
function dot$1(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) return multiply(x, y);
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2]
	});
}
/**
* Compute the tensor dot product of two N-dimensional arrays.
*
* The behavior is determined by `axes`. If an integer `k`, sum over the last
* `k` axes of x and the first `k` axes of y. If a tuple, then the first array
* corresponds to the axes of x and the second to the axes of y.
*/
function tensordot(x, y, axes = 2) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (typeof axes === "number") axes = [require_backend.range(-axes, 0), require_backend.range(axes)];
	return dot(x, y, {
		lhsContractingDims: axes[0],
		rhsContractingDims: axes[1]
	});
}
/**
* Einstein summation.
*
* This is a general API for performing tensor reductions, products,
* transpositions, and traces using Einstein notation for referring to named
* axes. See the docs for `numpy.einsum()` for more information.
*
* <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
*
* The full einsum API is implemented, including implicit and explicit output
* indices, ellipsis broadcasting, Unicode subscripts, and an optimal path
* ordering algorithm. It lowers to one or more calls to:
*
* - `jax.lax.dot()`
* - `jax.numpy.diagonal()`
* - `jax.numpy.sum()`
* - `jax.numpy.transpose()`
*/
function einsum(...args) {
	if (args.length === 0) throw new Error("einsum: must provide at least one argument");
	let input;
	let operands = [];
	if (typeof args[0] === "string") {
		operands = args.slice(1).map(fudgeArray);
		input = parseEinsumExpression(args[0], operands.map((x) => x.shape));
	} else {
		const n = args.length >> 1;
		const shapes = [];
		const lhsIndices = [];
		for (let i = 0; i < n; i++) {
			operands.push(fudgeArray(args[2 * i]));
			shapes.push(operands[i].shape);
			lhsIndices.push(args[2 * i + 1]);
		}
		let rhsIndex;
		if (args.length % 2 === 1) rhsIndex = args[2 * n];
		else {
			const indexCount = [];
			for (const i of lhsIndices.flat()) indexCount[i] = (indexCount[i] ?? 0) + 1;
			rhsIndex = [...indexCount.entries()].filter(([_, count]) => count === 1).map(([i, _]) => i);
		}
		input = {
			lhsIndices,
			rhsIndex,
			shapes
		};
	}
	const path = computeEinsumPath(input);
	if (require_backend.DEBUG >= 3) console.info(`einsum: computed path: ${path.approximateFlops} flops`);
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	const processSingleTensor = (ar$1, index$1, doNotReduce = []) => {
		index$1 = index$1.slice();
		diag: while (true) {
			for (let i = 0; i < index$1.length; i++) {
				const idx = index$1[i];
				const j = index$1.indexOf(idx, i + 1);
				if (j !== -1) {
					ar$1 = diagonal(ar$1, 0, i, j);
					index$1.splice(j, 1);
					index$1.splice(i, 1);
					index$1.push(idx);
					continue diag;
				}
			}
			break;
		}
		for (let i = index$1.length - 1; i >= 0; i--) {
			const idx = index$1[i];
			if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
				ar$1 = sum(ar$1, i);
				index$1.splice(i, 1);
			}
		}
		return [ar$1, index$1];
	};
	for (const [i, j] of path.path) {
		let indexReduced = [];
		const indexGroup = [];
		for (const idx of [...indices[i], ...indices[j]]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		const [a, aidx] = processSingleTensor(operands[i], indices[i], indices[j]);
		const [b, bidx] = processSingleTensor(operands[j], indices[j], indices[i]);
		indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
		const indexBatch = aidx.filter((idx) => bidx.includes(idx) && !indexReduced.includes(idx));
		const result = dot(a, b, {
			lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
			rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
			lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
			rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx))
		});
		operands.push(result);
		indices.push([
			...indexBatch,
			...aidx.filter((idx) => !bidx.includes(idx)),
			...bidx.filter((idx) => !aidx.includes(idx))
		]);
		for (const idx of indices[indices.length - 1]) ++indexUsageCounts[idx];
	}
	for (const idx of indices[operands.length - 1]) --indexUsageCounts[idx];
	const [ar, index] = processSingleTensor(operands[operands.length - 1], indices[operands.length - 1]);
	const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
	return ar.transpose(finalPerm);
}
/**
* Compute the inner product of two arrays.
*
* Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
* contraction on the last axis.
*
* Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
*/
function inner(x, y) {
	return dot(fudgeArray(x), fudgeArray(y), {
		lhsContractingDims: [-1],
		rhsContractingDims: [-1]
	});
}
/**
* Compute the outer product of two arrays.
*
* If the input arrays are not 1D, they will be flattened. Returned array will
* be of shape `[x.size, y.size]`.
*/
function outer(x, y) {
	x = ravel(x);
	y = ravel(y);
	return multiply(x.reshape([x.shape[0], 1]), y);
}
/** Vector dot product of two arrays along a given axis. */
function vecdot(x, y, { axis } = {}) {
	const xaxis = require_backend.checkAxis(axis ?? -1, ndim(x));
	const yaxis = require_backend.checkAxis(axis ?? -1, ndim(y));
	if (shape(x)[xaxis] !== shape(y)[yaxis]) throw new Error(`vecdot: shapes ${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`);
	x = moveaxis$1(x, xaxis, -1);
	y = moveaxis$1(y, yaxis, -1);
	return dot$2(x, y);
}
/**
* Return the dot product of two vectors.
*
* Like vecdot() but flattens the arguments first into vectors.
*/
function vdot(x, y) {
	return dot$2(ravel(x), ravel(y));
}
function _convImpl(name, x, y, mode) {
	if (x.ndim !== 1 || y.ndim !== 1) throw new Error(`${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`);
	let flipOutput = false;
	if (x.shape[0] < y.shape[0]) {
		[x, y] = [y, x];
		if (name === "correlate") flipOutput = true;
	}
	if (name === "convolve") y = flip(y);
	let padding;
	if (mode === "valid") padding = "VALID";
	else if (mode === "same") padding = "SAME_LOWER";
	else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
	else throw new Error(`${name}: invalid mode ${mode}, expected "full", "same", or "valid"`);
	const z = conv(x.slice(null, null), y.slice(null, null), [1], padding).slice(0, 0);
	return flipOutput ? flip(z) : z;
}
/** Convolution of two one-dimensional arrays. */
function convolve(x, y, mode = "full") {
	return _convImpl("convolve", x, y, mode);
}
/** Correlation of two one dimensional arrays. */
function correlate(x, y, mode = "valid") {
	return _convImpl("correlate", x, y, mode);
}
/**
* Return a tuple of coordinate matrices from coordinate vectors.
*
* Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
* fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
*/
function meshgrid(xs, { indexing } = {}) {
	indexing ??= "xy";
	for (const x of xs) if (x.ndim !== 1) throw new Error(`meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`);
	if (xs.length <= 1) return xs;
	if (indexing === "xy") {
		const [a, b, ...rest] = xs;
		const [rb, ra, ...rrest] = meshgrid([
			b,
			a,
			...rest
		], { indexing: "ij" });
		return [
			ra,
			rb,
			...rrest
		];
	}
	const shape$1 = xs.map((x) => x.shape[0]);
	return xs.map((x, i) => broadcast(x, shape$1, [...require_backend.range(i), ...require_backend.range(i + 1, xs.length)]));
}
/**
* Clip (limit) the values in an array.
*
* Given an interval, values outside the interval are clipped to the interval
* edges. For example, if an interval of [0, 1] is specified, values smaller
* than 0 become 0, and values larger than 1 become 1.
*
* If either bound is undefined, it is ignored.
*/
function clip(a, min$2, max$2) {
	a = fudgeArray(a);
	if (max$2 !== void 0) a = minimum(a, max$2);
	if (min$2 !== void 0) a = maximum(a, min$2);
	return a;
}
/**
* Calculate the absolute value element-wise.
*
* This is the same function as `jax.numpy.abs()`.
*/
function absolute(x) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), x.ref.mul(-1), x);
}
/** Return an element-wise indication of sign of the input. */
function sign(x) {
	x = fudgeArray(x);
	return where(notEqual(x.ref, 0), where(less(x.ref, 0), -1, 1), 0);
}
/** @function Return element-wise positive values of the input (no-op). */
const positive = fudgeArray;
/**
* Return the Hamming window of size M, a taper with a weighted cosine bell.
*
* `w(n) = 0.54 - 0.46 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
*/
function hamming(M) {
	return cos(linspace(0, 2 * Math.PI, M)).mul(-.46).add(.54);
}
/**
* Return the Hann window of size M, a taper with a weighted cosine bell.
*
* `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
*/
function hann(M) {
	return cos(linspace(0, 2 * Math.PI, M)).mul(-.5).add(.5);
}
/**
* @function
* Compute the Heaviside step function. It is defined piecewise:
* - `heaviside(x1, x2) = 0` for `x1 < 0`,
* - `heaviside(x1, x2) = x2` for `x1 == 0`,
* - `heaviside(x1, x2) = 1` for `x1 > 0`.
*/
const heaviside = jit$1(function heaviside$1(x1, x2) {
	return where(less(x1.ref, 0), 0, where(equal(x1, 0), x2, 1));
});
/** Calculate element-wise square of the input array. */
function square(x) {
	x = fudgeArray(x);
	return x.ref.mul(x);
}
/** Element-wise tangent function (takes radians). */
function tan(x) {
	x = fudgeArray(x);
	return sin(x.ref).div(cos(x));
}
/**
* @function
* Return the normalized sinc function.
*
* The sinc function is defined as `sin(πx) / (πx)` for `x != 0`, and `1` for `x = 0`.
* This is the normalized sinc function commonly used in signal processing.
*
* **Note:** JVP is not supported at x=0 due to discontinuous derivative. This
* requires a custom JVP rule to handle properly (see JAX implementation).
*/
const sinc = jit$1(function sinc$1(x) {
	const pix = x.ref.mul(Math.PI);
	return where(equal(x, 0), 1, sin(pix.ref).div(pix));
});
/** Element-wise inverse cosine function (inverse of cos). */
function acos(x) {
	return subtract(pi / 2, asin(x));
}
/**
* @function
* Return element-wise hypotenuse for the given legs of a right triangle.
*
* In the original NumPy/JAX implementation, this function is more numerically
* stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
* stability improvements.
*/
const hypot = jit$1(function hypot$1(x1, x2) {
	return sqrt(square(x1).add(square(x2)));
});
/**
* @function
* Element-wise arc tangent of y/x with correct quadrant.
*
* Returns the angle in radians between the positive x-axis and the point (x, y).
* The result is in the range [-π, π].
*
* Uses numerically stable formulas:
* - When x >= 0: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
* - When x < 0:  atan2(y, x) = 2 * atan((sqrt(x^2 + y^2) - x) / y)
*
* The output is ill-defined when both x and y are zero.
*/
const atan2 = jit$1(function atan2$1(y, x) {
	const r = sqrt(square(x.ref).add(square(y.ref)));
	const xNeg = less(x.ref, 0);
	const numer = where(xNeg.ref, r.ref.sub(x.ref), y.ref);
	const denom = where(xNeg, y, r.add(x));
	return atan(numer.div(denom)).mul(2);
});
/** Element-wise subtraction, with broadcasting. */
function subtract(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	return x.sub(y);
}
/** Calculates the floating-point division of x by y element-wise. */
function trueDivide(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) {
		x = x.astype(require_backend.DType.Float32);
		y = y.astype(require_backend.DType.Float32);
	}
	return x.div(y);
}
/**
* Return the largest integer smaller or equal to the division of the inputs.
*
* The result is always rounded towards negative infinity.
*
* For floating-point inputs, this is equivalent to `floor(x / y)`.
* For integer inputs, we use `(x - remainder(x, y)) / y` to handle
* negative values correctly (note: may overflow near int32 boundaries).
*
* @param x - Dividend array.
* @param y - Divisor array.
* @returns Element-wise floor division of x by y.
*/
function floorDivide(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (require_backend.isFloatDtype(x.dtype) || require_backend.isFloatDtype(y.dtype)) return floor(trueDivide(x, y));
	return subtract(x, remainder(x.ref, y.ref)).div(y);
}
/**
* @function
* Calculate element-wise floating-point modulo operation.
*/
const fmod = jit$1(function fmod$1(x, y) {
	return x.ref.sub(y.ref.mul(idiv(x, y)));
});
/**
* @function
* Calculate element-wise remainder of the division (matches sign of y).
*/
const remainder = jit$1(function remainder$1(x, y) {
	return mod(mod(x, y.ref).add(y.ref), y);
});
/**
* Return element-wise quotient and remainder simultaneously.
*
* Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
*
* @param x - Dividend array.
* @param y - Divisor array.
* @returns Tuple of [quotient, remainder].
*/
function divmod(x, y) {
	const xArr = fudgeArray(x);
	const yArr = fudgeArray(y);
	return [floorDivide(xArr.ref, yArr.ref), remainder(xArr, yArr)];
}
/** Round input to the nearest integer towards zero. */
function trunc(x) {
	return idiv(x, 1);
}
/**
* Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
*
* This is the inverse of `frexp()`.
*/
function ldexp(x1, x2) {
	return multiply(x1, exp2(x2));
}
/**
* Decompose floating-point values into mantissa and two's exponent.
*
* The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
* `x != 0`, and the exponent is an integer such that
* `x = mantissa * 2**exponent`.
*/
function frexp(x) {
	x = fudgeArray(x);
	const absx = absolute(x.ref);
	const exponent = where(equal(x.ref, 0), 0, floor(log2(absx)).add(1).astype(require_backend.DType.Int32));
	const mantissa = x.div(exp2(exponent.ref.astype(x.dtype)));
	return [mantissa, exponent];
}
/** Calculate `2**p` for all p in the input array. */
function exp2(p) {
	return exp(multiply(p, Math.LN2));
}
/** Return the base-2 logarithm of x, element-wise. */
function log2(x) {
	return log(x).mul(Math.LOG2E);
}
/** Return the base-10 logarithm of x, element-wise. */
function log10(x) {
	return log(x).mul(Math.LOG10E);
}
/** Calculate `exp(x) - 1` element-wise. */
function expm1(x) {
	return exp(x).sub(1);
}
/** Calculate the natural logarithm of `1 + x` element-wise. */
function log1p(x) {
	return log(add(1, x));
}
/** Convert angles from degrees to radians. */
function deg2rad(x) {
	return multiply(x, pi / 180);
}
/** @function Alias of `jax.numpy.deg2rad()`. */
const radians = deg2rad;
/** Convert angles from radians to degrees. */
function rad2deg(x) {
	return multiply(x, 180 / pi);
}
/** @function Alias of `jax.numpy.rad2deg()`. */
const degrees = rad2deg;
/**
* @function
* Computes first array raised to power of second array, element-wise.
*/
const power = jit$1(function power$1(x1, x2) {
	const x2i = trunc(x2.ref);
	const shouldBeNaN = multiply(x2.ref.notEqual(x2i.ref), x1.ref.less(0));
	const resultSign = where(mod(x2i, 2).notEqual(0), where(x1.ref.less(0), -1, 1), 1);
	return where(shouldBeNaN, nan, exp(log(absolute(x1)).mul(x2)).mul(resultSign));
});
/** @function Calculate the element-wise cube root of the input array. */
const cbrt = jit$1(function cbrt$1(x) {
	const sgn = where(less(x.ref, 0), -1, 1);
	return sgn.ref.mul(exp(log(x.mul(sgn)).mul(1 / 3)));
});
/**
* @function
* Calculate element-wise hyperbolic sine of input.
*
* `sinh(x) = (exp(x) - exp(-x)) / 2`
*/
const sinh = jit$1(function sinh$1(x) {
	const ex = exp(x);
	const emx = reciprocal(ex.ref);
	return ex.sub(emx).mul(.5);
});
/**
* @function
* Calculate element-wise hyperbolic cosine of input.
*
* `cosh(x) = (exp(x) + exp(-x)) / 2`
*/
const cosh = jit$1(function cosh$1(x) {
	const ex = exp(x);
	const emx = reciprocal(ex.ref);
	return ex.add(emx).mul(.5);
});
/**
* @function
* Calculate element-wise hyperbolic tangent of input.
*
* `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
*/
const tanh = jit$1(function tanh$1(x) {
	const negsgn = where(less(x.ref, 0), 1, -1);
	const en2x = exp(x.mul(negsgn.ref).mul(2));
	return en2x.ref.sub(1).div(en2x.add(1)).mul(negsgn);
});
/**
* @function
* Calculate element-wise inverse hyperbolic sine of input.
*
* `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
*/
const arcsinh = jit$1(function arcsinh$1(x) {
	return log(x.ref.add(sqrt(square(x).add(1))));
});
/**
* @function
* Calculate element-wise inverse hyperbolic cosine of input.
*
* `arccosh(x) = ln(x + sqrt(x^2 - 1))`
*/
const arccosh = jit$1(function arccosh$1(x) {
	return log(x.ref.add(sqrt(square(x).sub(1))));
});
/**
* @function
* Calculate element-wise inverse hyperbolic tangent of input.
*
* `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
*/
const arctanh = jit$1(function arctanh$1(x) {
	return log(add(1, x.ref).div(subtract(1, x))).mul(.5);
});
/**
* Compute the variance of an array.
*
* The variance is computed for the flattened array by default, otherwise over
* the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function var_(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
	if (n === 0) throw new Error("var: cannot compute variance over zero-length axis");
	const mu = opts?.mean !== void 0 ? opts.mean : mean(x.ref, axis, { keepdims: true });
	return square(x.sub(mu)).sum(axis, { keepdims: opts?.keepdims }).mul(1 / (n - (opts?.correction ?? 0)));
}
/**
* Compute the standard deviation of an array.
*
* The standard deviation is computed for the flattened array by default,
* otherwise over the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function std(x, axis = null, opts) {
	return sqrt(var_(x, axis, opts));
}
/** Estimate the sample covariance of a set of variables. */
function cov(x, y = null, { rowvar = true } = {}) {
	x = fudgeArray(x);
	if (x.ndim === 1) x = x.reshape([1, x.shape[0]]);
	if (y !== null) {
		y = fudgeArray(y);
		if (y.ndim === 1) y = y.reshape([1, y.shape[0]]);
		x = vstack([x, y]);
	}
	if (!rowvar) x = x.transpose();
	const [_M, N] = x.shape;
	x = x.ref.sub(x.mean(1, { keepdims: true }));
	return dot$1(x.ref, x.transpose()).div(N - 1);
}
/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
function corrcoef(x, y) {
	const c = cov(x, y);
	const variances = diag(c.ref);
	const norm = sqrt(outer(variances.ref, variances));
	return c.div(norm);
}
/** Test element-wise for positive or negative infinity, return bool array. */
function isinf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.ref.equal(Infinity).add(x.equal(-Infinity)) : fullLike$1(x, false);
}
/** Test element-wise for NaN (Not a Number). */
function isnan(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.ref.notEqual(x) : fullLike$1(x, false);
}
/** Test element-wise for negative infinity, return bool array. */
function isneginf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike$1(x, false);
}
/** Test element-wise for positive infinity, return bool array. */
function isposinf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike$1(x, false);
}
/**
* Replace NaN and infinite entries in an array.
*
* By default, NaNs are replaced with `0.0`, and infinities are are substituted
* with the corresponding maximum or minimum finite values.
*/
function nanToNum(x, { nan: nan$1 = 0, posinf = null, neginf = null } = {}) {
	x = fudgeArray(x);
	x = where(isnan(x.ref), nan$1, x);
	posinf ??= require_backend.isFloatDtype(x.dtype) ? finfo(x.dtype).max : iinfo(x.dtype).max;
	neginf ??= require_backend.isFloatDtype(x.dtype) ? finfo(x.dtype).min : iinfo(x.dtype).min;
	x = where(isposinf(x.ref), posinf, x);
	x = where(isneginf(x.ref), neginf, x);
	return x;
}
/**
* @function
* Test element-wise for finite values (not infinity or NaN).
*/
const isfinite = jit$1(function isfinite$1(x) {
	if (!require_backend.isFloatDtype(x.dtype)) return fullLike$1(x, true);
	return isnan(x.ref).add(isinf(x)).notEqual(true);
});

//#endregion
//#region src/library/lax-linalg.ts
var lax_linalg_exports = {};
__export(lax_linalg_exports, {
	cholesky: () => cholesky$1,
	lu: () => lu,
	triangularSolve: () => triangularSolve
});
/**
* Compute the Cholesky decomposition of a symmetric positive-definite matrix.
*
* The Cholesky decomposition of a matrix `A` is:
*
* - A = L @ L^T  (for upper=false, default)
* - A = U^T @ U  (for upper=true)
*
* where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
* The input matrix must be symmetric and positive-definite.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const x = np.array([[2., 1.], [1., 2.]]);
*
* // Lower Cholesky factorization (default):
* const L = lax.linalg.cholesky(x);
* // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
*
* // Upper Cholesky factorization:
* const U = lax.linalg.cholesky(x, { upper: true });
* // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
* ```
*/
function cholesky$1(a, { upper = false } = {}) {
	const L = cholesky$2(a);
	return upper ? moveaxis$1(L, -2, -1) : L;
}
/**
* LU decomposition with partial pivoting.
*
* Computes the matrix decomposition: `P @ A = L @ U`, where `P` is a
* permutation of the rows of `A`, `L` is lower-triangular with unit diagonal,
* and `U` is upper-triangular.
*
* @param x - A batch of matrices with shape `[..., m, n]`.
*
* @returns A tuple `(lu, pivots, permutation)` where:
* - `lu`: combined lower and upper triangular matrices.
* - `pivots`: an array of pivot indices with shape `[..., min(m, n)]`.
* - `permutation`: the permutation generated by pivots with shape `[..., m]`.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const A = np.array([[4., 3.], [6., 3.]]);
* const [lu, pivots, permutation] = lax.linalg.lu(A);
* // lu ≈ [[6., 3.], [0.6666667, 1.0]]
* // pivots = [1, 1]
* // permutation = [1, 0]
* ```
*/
function lu(x) {
	return lu$1(x);
}
/**
* Solve a triangular linear system.
*
* Solves `a @ x = b` (if leftSide=true) or `x @ a = b` (if leftSide=false)
* where `a` is a triangular matrix.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const L = np.array([[2., 0.], [1., 3.]]);
* const b = np.array([4., 7.]).reshape([2, 1]);
*
* // Solve L @ x = b
* const x = lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true });
* // x = [[2.], [5./3.]]
* ```
*/
function triangularSolve(a, b, { leftSide = false, lower = false, transposeA = false, unitDiagonal = false } = {}) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	if (!leftSide) transposeA = !transposeA;
	else b = moveaxis$1(b, -2, -1);
	if (transposeA) a = moveaxis$1(a, -2, -1);
	let x = triangularSolve$1(a, b, {
		lower,
		unitDiagonal
	});
	if (leftSide) x = moveaxis$1(x, -2, -1);
	return x;
}

//#endregion
//#region src/library/lax-scan.ts
function scan(f, init$1, xs, options) {
	const opts = typeof options === "number" ? { length: options } : options ?? {};
	const { length: lengthOpt, reverse = false, requirePath } = opts;
	const [initFlat, initTreedef] = flatten(init$1);
	const [xsFlat, xsTreedef] = flatten(xs);
	const n = lengthOpt ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);
	if (n === 0) throw new Error("scan: cannot determine length from empty inputs");
	const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
	const xSliceAvals = xsFlat.map((arr) => {
		const aval = getAval(arr);
		return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
	});
	const flatF = (carryFlat, xSliceFlat) => {
		const carry = unflatten(initTreedef, carryFlat);
		const xSlice = unflatten(xsTreedef, xSliceFlat);
		const [newCarry, y] = f(carry, xSlice);
		const [newCarryFlat] = flatten(newCarry);
		const [yFlat, yTreedef$1] = flatten(y);
		flatF._yTreedef = yTreedef$1;
		return [newCarryFlat, yFlat];
	};
	const traceFn = (...args) => {
		const numCarry$1 = carryAvals.length;
		const carryFlat = args.slice(0, numCarry$1);
		const xSliceFlat = args.slice(numCarry$1);
		const [newCarryFlat, yFlat] = flatF(carryFlat, xSliceFlat);
		return [...newCarryFlat, ...yFlat];
	};
	const traceAvals = [...carryAvals, ...xSliceAvals];
	const { jaxpr: closedJaxpr, treedef: _outTreedef } = makeJaxpr$1(traceFn)(...traceAvals);
	const jaxpr = closedJaxpr.jaxpr;
	const consts = closedJaxpr.consts;
	const scanArgs = [
		...consts.map((c) => c.ref),
		...initFlat.map((arr) => arr.ref),
		...xsFlat.map((arr) => arr.ref)
	];
	const numCarry = initFlat.length;
	const numConsts = consts.length;
	const results = bind(Primitive.Scan, scanArgs, {
		jaxpr,
		numCarry,
		numConsts,
		length: n,
		reverse,
		requirePath
	});
	initFlat.forEach((arr) => arr.dispose());
	xsFlat.forEach((arr) => arr.dispose());
	closedJaxpr.dispose();
	const carryOut = results.slice(0, numCarry);
	const ysFlat = results.slice(numCarry);
	const finalCarry = unflatten(initTreedef, carryOut);
	const yTreedef = flatF._yTreedef;
	const ys = unflatten(yTreedef, ysFlat);
	return [finalCarry, ys];
}
/**
* Stack a list of pytrees along a new leading axis.
* Each pytree in the list must have the same structure.
*/
function stackPyTree(trees) {
	if (trees.length === 0) throw new Error("stackPyTree: empty list");
	const [firstLeaves, treedef] = flatten(trees[0]);
	const allLeaves = trees.map((t) => leaves(t));
	const numLeaves = firstLeaves.length;
	const stackedLeaves = [];
	for (let leafIdx = 0; leafIdx < numLeaves; leafIdx++) {
		const toStack = allLeaves.map((leaves$1) => leaves$1[leafIdx].ref);
		const stacked = stack(toStack, 0);
		stackedLeaves.push(stacked);
	}
	return unflatten(treedef, stackedLeaves);
}

//#endregion
//#region src/library/lax.ts
var lax_exports = {};
__export(lax_exports, {
	conv: () => conv,
	convGeneralDilated: () => convGeneralDilated,
	convTranspose: () => convTranspose,
	convWithGeneralPadding: () => convWithGeneralPadding,
	dot: () => dot,
	erf: () => erf,
	erfc: () => erfc,
	linalg: () => lax_linalg_exports,
	reduceWindow: () => reduceWindow,
	scan: () => scan,
	stackPyTree: () => stackPyTree,
	stopGradient: () => stopGradient$1
});
const JsArray = globalThis.Array;
/**
* General dot product/contraction operator.
*
* Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
* `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
*/
function dot(lhs, rhs, { lhsContractingDims: lc = [], rhsContractingDims: rc = [], lhsBatchDims: lb = [], rhsBatchDims: rb = [] } = {}) {
	if (lc.length !== rc.length) throw new Error(`dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`);
	else if (lb.length !== rb.length) throw new Error(`dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`);
	lc = lc.map((a) => require_backend.checkAxis(a, lhs.ndim));
	rc = rc.map((a) => require_backend.checkAxis(a, rhs.ndim));
	lb = lb.map((a) => require_backend.checkAxis(a, lhs.ndim));
	rb = rb.map((a) => require_backend.checkAxis(a, rhs.ndim));
	if (lc.some((a) => lb.includes(a))) throw new Error(`dot: lhs contracting dims ${JSON.stringify(lc)} overlap with batch dims ${JSON.stringify(lb)}`);
	else if (rc.some((a) => rb.includes(a))) throw new Error(`dot: rhs contracting dims ${JSON.stringify(rc)} overlap with batch dims ${JSON.stringify(rb)}`);
	const lf = require_backend.range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
	const rf = require_backend.range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
	const lhs2 = lhs.transpose([
		...lb,
		...lf,
		...lc
	]);
	const rhs2 = rhs.transpose([
		...rb,
		...rf,
		...rc
	]);
	if (lc.length === 0) return mul(lhs2.reshape([
		...lb.map((a) => lhs.shape[a]),
		...lf.map((a) => lhs.shape[a]),
		...require_backend.rep(rf.length, 1)
	]), rhs2.reshape([
		...rb.map((a) => rhs.shape[a]),
		...require_backend.rep(lf.length, 1),
		...rf.map((a) => rhs.shape[a])
	]));
	const dotShapeX = lc.map((a) => lhs.shape[a]);
	const dotShapeY = rc.map((a) => rhs.shape[a]);
	if (!require_backend.deepEqual(dotShapeX, dotShapeY)) throw new Error(`dot: shapes not aligned along contracting dims: ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`);
	return dot$2(lhs2.reshape([
		...lb.map((a) => lhs.shape[a]),
		...lf.map((a) => lhs.shape[a]),
		...require_backend.rep(rf.length, 1),
		require_backend.prod(dotShapeX)
	]), rhs2.reshape([
		...rb.map((a) => rhs.shape[a]),
		...require_backend.rep(lf.length, 1),
		...rf.map((a) => rhs.shape[a]),
		require_backend.prod(dotShapeY)
	]));
}
function padtypeToPads(inShape, filterShape, strides, dilation, padding) {
	const padType = padding.toUpperCase();
	switch (padType) {
		case "VALID": return require_backend.rep(inShape.length, [0, 0]);
		case "SAME":
		case "SAME_LOWER": {
			const outShape = inShape.map((size$1, i) => Math.ceil(size$1 / strides[i]));
			const padSizes = require_backend.zipn(outShape, strides, filterShape, dilation, inShape).map(([o, s, k, d, i]) => Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i));
			if (padType === "SAME") return padSizes.map((size$1) => [size$1 >> 1, size$1 - (size$1 >> 1)]);
			else return padSizes.map((size$1) => [size$1 - (size$1 >> 1), size$1 >> 1]);
		}
		default: throw new Error(`Unknown padding type: ${padType}`);
	}
}
/**
* General n-dimensional convolution operator, with optional dilation.
*
* The semantics of this operation mimic the `jax.lax.conv_general_dilated`
* function in JAX, which wraps XLA's general convolution operator.
*
* @param lhs - Input tensor; shape `[N, C_in, ...xs]`
* @param rhs - Convolution kernel; shape `[C_out, C_in / G, ...ks]`
* @param windowStrides - Strides for each spatial dimension
* @param padding - Padding for each spatial dimension, or a string
*   (`"VALID"`, `"SAME"`, or `"SAME_LOWER"`)
*/
function convGeneralDilated(lhs, rhs, windowStrides, padding, { lhsDilation, rhsDilation, featureGroupCount = 1 } = {}) {
	if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
	if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
	if (typeof padding === "string") {
		if (lhsDilation?.some((d) => d !== 1)) throw new Error("String padding is not supported for transposed convolutions");
		padding = padtypeToPads(lhs.shape.slice(2), rhs.shape.slice(2), windowStrides, rhsDilation ?? require_backend.rep(rhs.ndim - 2, 1), padding);
	}
	if (featureGroupCount !== 1) {
		const G = featureGroupCount;
		const [N, C_in, ...xs] = lhs.shape;
		const [C_out, C_in_per_group, ...ks] = rhs.shape;
		if (C_in % G !== 0) throw new Error(`featureGroupCount=${G} must divide input channels=${C_in}`);
		if (C_out % G !== 0) throw new Error(`featureGroupCount=${G} must divide output channels=${C_out}`);
		if (C_in / G !== C_in_per_group) throw new Error(`rhs input channels=${C_in_per_group} must equal lhs input channels / groups=${C_in / G}`);
		const lhsGrouped = moveaxis(lhs.reshape([
			N,
			G,
			C_in / G,
			...xs
		]), 1, 0);
		const rhsGrouped = rhs.reshape([
			G,
			C_out / G,
			C_in_per_group,
			...ks
		]);
		const result = conv$1(lhsGrouped, rhsGrouped, {
			vmapDims: 1,
			strides: windowStrides,
			padding,
			lhsDilation,
			rhsDilation
		});
		const ys = result.shape.slice(3);
		return moveaxis(result, 0, 1).reshape([
			N,
			C_out,
			...ys
		]);
	}
	return conv$1(lhs, rhs, {
		strides: windowStrides,
		padding,
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function convWithGeneralPadding(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding, {
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function conv(lhs, rhs, windowStrides, padding) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding);
}
/**
* Convenience wrapper for calculating the N-d convolution "transpose".
*
* This function directly calculates a fractionally strided conv rather than
* indirectly calculating the gradient (transpose) of a forward convolution.
* It is equivalent to the JAX version, except:
*
* - The `use_consistent_padding` option is not available. We only have the
*   consistent padding case (JAX version >0.8.4).
* - The order of dimensions matches `lax.conv_general_dilated`.
*
* Unlike PyTorch/TensorFlow, by default we don't reverse the kernel's spatial
* dimensions or the `(C_out, C_in)` axis order. To get this behavior, set
* `transposeKernel` to true.
*
* @param lhs - Input tensor; shape `[N, C_in, ...xs]`
* @param rhs - Convolution kernel; shape `[C_out, C_in, ...ks]`
* @param strides - Sequence of n integers, sets fractional stride
* @param padding - Apply padding of `dilation * (kernel_size - 1) - padding` to
*   each side of the input, so it acts like gradient of `conv()`
* @param rhsDilation - Atrous dilation for the kernel
* @param transposeKernel - Flip spatial axes and swap the input/output channels
*   of the kernel; its shape should be `[C_in, C_out, ...ks]`
*/
function convTranspose(lhs, rhs, strides, padding, { rhsDilation, transposeKernel = false } = {}) {
	const kernelShape = rhs.shape.slice(2);
	rhsDilation = rhsDilation ?? require_backend.rep(kernelShape.length, 1);
	const effectiveKernel = kernelShape.map((k, i) => Math.max(0, (k - 1) * rhsDilation[i] + 1));
	const pads = effectiveKernel.map((k, i) => convTransposePadding(k, strides[i], typeof padding === "string" ? padding : padding[i]));
	if (transposeKernel) {
		rhs = flip$1(rhs, require_backend.range(2, rhs.ndim));
		rhs = moveaxis(rhs, 0, 1);
	}
	return convGeneralDilated(lhs, rhs, require_backend.rep(lhs.ndim - 2, 1), pads, {
		lhsDilation: strides,
		rhsDilation
	});
}
function convTransposePadding(k, s, padding) {
	let padLen;
	let pad1;
	if (padding === "SAME") {
		padLen = k + s - 2;
		pad1 = s > k - 1 ? k - 1 : Math.ceil(padLen / 2);
	} else if (padding === "VALID") {
		padLen = k + s - 2 + Math.max(k - s, 0);
		pad1 = k - 1;
	} else if (JsArray.isArray(padding)) {
		const pads = [k - 1 - padding[0], k - 1 - padding[1]];
		pad1 = pads[0];
		padLen = pads[0] + pads[1];
	} else throw new Error(`convTranspose: Invalid padding type ${padding}`);
	return [pad1, padLen - pad1];
}
/** Reduce a computation over padded windows. */
function reduceWindow(operand, computation, windowDimensions, windowStrides) {
	if (operand.ndim < windowDimensions.length) throw new Error(`Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`);
	if (!windowStrides) windowStrides = require_backend.rep(windowDimensions.length, 1);
	for (let i = 0; i < operand.ndim; i++) computation = vmap$1(computation, 0);
	return computation(bind1(Primitive.Pool, [operand], {
		window: windowDimensions,
		strides: windowStrides
	}));
}
/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
function erf(x) {
	return erf$1(x);
}
/**
* The complementary error function: `erfc(x) = 1 - erf(x)`.
*
* This function is more accurate than `1 - erf(x)` for large values of `x`,
* where `erf(x)` is very close to 1.
*/
function erfc(x) {
	return erfc$1(x);
}
/**
* Stops gradient computation.
*
* Behaves as the identity function but prevents the flow of gradients during
* forward or reverse-mode automatic differentiation.
*/
function stopGradient$1(x) {
	return stopGradient(x);
}

//#endregion
//#region src/library/nn.ts
var nn_exports = {};
__export(nn_exports, {
	celu: () => celu,
	dotProductAttention: () => dotProductAttention,
	elu: () => elu,
	gelu: () => gelu,
	glu: () => glu,
	hardSigmoid: () => hardSigmoid,
	hardSilu: () => hardSilu,
	hardSwish: () => hardSilu,
	hardTanh: () => hardTanh,
	identity: () => identity,
	leakyRelu: () => leakyRelu,
	logSigmoid: () => logSigmoid,
	logSoftmax: () => logSoftmax,
	logmeanexp: () => logmeanexp,
	logsumexp: () => logsumexp,
	mish: () => mish,
	oneHot: () => oneHot,
	relu: () => relu,
	relu6: () => relu6,
	selu: () => selu,
	sigmoid: () => sigmoid,
	silu: () => silu,
	softSign: () => softSign,
	softmax: () => softmax,
	softplus: () => softplus,
	sparsePlus: () => sparsePlus,
	sparseSigmoid: () => sparseSigmoid,
	squareplus: () => squareplus,
	standardize: () => standardize,
	swish: () => silu
});
/**
* Rectified Linear Unit (ReLU) activation function:
* `relu(x) = max(x, 0)`.
*/
function relu(x) {
	return maximum(x, 0);
}
/**
* Rectified Linear Unit 6 (ReLU6) activation function:
* `relu6(x) = min(max(x, 0), 6)`.
*/
function relu6(x) {
	return clip(x, 0, 6);
}
/**
* Sigmoid activation function, computed element-wise:
* `sigmoid(x) = 1 / (1 + exp(-x))`.
*
* Reference: https://en.wikipedia.org/wiki/Sigmoid_function
*/
function sigmoid(x) {
	return reciprocal(exp(negative(x)).add(1));
}
/**
* Softplus activation function:
* `softplus(x) = log(1 + exp(x))`.
*
* Reference: https://en.wikipedia.org/wiki/Softplus
*/
function softplus(x) {
	return log(exp(x).add(1));
}
/**
* @function
* Sparse plus function:
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x+1)**2 / 4`
* - When `x >= 1`: `x`
*/
const sparsePlus = jit$1((x) => {
	return where(x.ref.lessEqual(-1), 0, where(x.ref.less(1), square(x.ref.add(1)).mul(.25), x));
});
/**
* @function
* Sparse sigmoid activation function.
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x + 1) / 2`
* - When `x >= 1`: `1`
*/
const sparseSigmoid = jit$1((x) => {
	return clip(x.add(1).mul(.5), 0, 1);
});
/**
* Soft-sign activation function, computed element-wise:
* `softsign(x) = x / (|x| + 1)`.
*/
function softSign(x) {
	x = fudgeArray(x);
	return x.ref.div(absolute(x).add(1));
}
/**
* @function
* Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
* Swish, computed element-wise:
* `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
*
* `swish()` and `silu()` are both aliases for the same function.
*
* Reference: https://en.wikipedia.org/wiki/Swish_function
*/
const silu = jit$1(function silu$1(x) {
	return x.ref.mul(sigmoid(x));
});
/**
* Log-sigmoid activation function, computed element-wise:
* `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
*/
function logSigmoid(x) {
	return negative(softplus(negative(x)));
}
/**
* @function
* Identity activation function. Returns the argument unmodified.
*/
const identity = fudgeArray;
/** Leaky rectified linear (ReLU) activation function */
function leakyRelu(x, negativeSlope = .01) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), x.ref.mul(negativeSlope), x);
}
/** Hard sigmoid activation function: `relu6(x+3)/6`. */
function hardSigmoid(x) {
	return relu6(add(x, 3)).mul(1 / 6);
}
/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
function hardSilu(x) {
	x = fudgeArray(x);
	return x.ref.mul(hardSigmoid(x));
}
/** Hard tanh activation function: `clip(x, -1, 1)`. */
function hardTanh(x) {
	return clip(x, -1, 1);
}
/**
* Exponential linear unit activation function.
*
* Computes the element-wise function:
* `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
*/
function elu(x, alpha = 1) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), exp(x.ref).sub(1).mul(alpha), x);
}
/**
* Continuously-differentiable exponential linear unit activation function.
*
* Computes the element-wise function:
* `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
*/
function celu(x, alpha = 1) {
	x = fudgeArray(x);
	return where(less(x.ref, 0), exp(x.ref.div(alpha)).sub(1).mul(alpha), x);
}
/**
* @function
* Scaled exponential linear unit activation.
*
* Computes the element-wise function:
* `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
*
* Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
*/
const selu = jit$1(function selu$1(x) {
	const alpha = 1.6732632423543772;
	const lambda = 1.0507009873554805;
	return where(x.ref.less(0), expm1(x.ref).mul(alpha), x).mul(lambda);
});
/**
* @function
* Gaussion error linear unit (GELU) activation function.
*
* This is computed element-wise. There are two variants depending on whether
* `approximate` is set (default true):
*
* - Approximate: `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
* - Exact: `gelu(x) = x * 0.5 * erfc(-x / sqrt(2))`
*
* Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
*/
const gelu = jit$1(function gelu$1(x, opts) {
	if (opts?.approximate ?? true) {
		const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
		return x.ref.mul(.5).mul(tanh(x.ref.mul(x.ref.mul(x).mul(.044715).add(1)).mul(SQRT_2_OVER_PI)).add(1));
	} else return x.ref.mul(.5).mul(erfc$1(negative(x.ref.mul(Math.SQRT1_2))));
}, { staticArgnums: [1] });
/**
* Gated linear unit (GLU) activation function.
*
* Splits the `axis` dimension of the input into two halves, a and b, then
* computes `a * sigmoid(b)`.
*/
function glu(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.checkAxis(axis, x.ndim);
	const size$1 = x.shape[axis];
	if (size$1 % 2 !== 0) throw new Error(`glu: axis ${axis} of shape (${x.shape}) does not have even length`);
	const slice = x.shape.map((a$1) => [0, a$1]);
	const a = shrink(x.ref, slice.toSpliced(axis, 1, [0, size$1 / 2]));
	const b = shrink(x, slice.toSpliced(axis, 1, [size$1 / 2, size$1]));
	return a.mul(sigmoid(b));
}
/**
* Squareplus activation function.
*
* Computes the element-wise function:
* `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
*/
function squareplus(x, b = 4) {
	x = fudgeArray(x);
	return x.ref.add(sqrt(square(x).add(b))).mul(.5);
}
/**
* Mish activation function.
*
* Computes the element-wise function:
* `mish(x) = x * tanh(softplus(x))`
*/
function mish(x) {
	x = fudgeArray(x);
	return x.ref.mul(tanh(softplus(x)));
}
/**
* Softmax function. Computes the function which rescales elements to the range
* [0, 1] such that the elements along `axis` sum to 1.
*
* If `axis` is not specified, it defaults to the last axis.
*
* Reference: https://en.wikipedia.org/wiki/Softmax_function
*/
function softmax(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return onesLike(x);
	const xMax = max(x.ref, axis, { keepdims: true });
	const unnormalized = exp(x.sub(stopGradient(xMax)));
	return unnormalized.ref.div(unnormalized.sum(axis, { keepdims: true }));
}
/**
* Log-Softmax function.
*
* Computes the logarithm of the `softmax` function, which rescales elements to
* the range [-infinity, 0).
*
* If `axis` is not specified, it defaults to the last axis.
*/
function logSoftmax(x, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return zerosLike(x);
	const xMax = max(x.ref, axis, { keepdims: true });
	const shifted = x.sub(stopGradient(xMax));
	const shiftedLogsumexp = log(exp(shifted.ref).sum(axis, { keepdims: true }));
	return shifted.sub(shiftedLogsumexp);
}
/**
* Log-sum-exp reduction. Also a multivariate version of `softplus`.
*
* If no axis is specified, the reduction is performed over all elements. This
* convention differs from `jax.nn.logSoftmax()`.
*
* Reference: https://en.wikipedia.org/wiki/LogSumExp
*/
function logsumexp(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const xMax = stopGradient(max(x.ref, axis, { keepdims: true }));
	const shifted = x.sub(xMax.ref);
	const result = xMax.add(log(exp(shifted).sum(axis, { keepdims: true })));
	return opts?.keepdims ? result : squeeze(result, axis);
}
/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
function logmeanexp(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
	return logsumexp(x, axis, opts).sub(Math.log(n));
}
/**
* Standardizes input to zero mean and unit variance.
*
* By default, this is computed over the last axis. You can pass in a different
* axis, or `null` to standardize over all elements.
*
* Epsilon is added to denominator, it defaults to `1e-5` for stability.
*/
function standardize(x, axis = -1, opts = {}) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const mu = opts.mean !== void 0 ? fudgeArray(opts.mean) : x.ref.mean(axis, { keepdims: true });
	const sigma2 = opts.variance !== void 0 ? fudgeArray(opts.variance) : square(x.ref).mean(axis, { keepdims: true }).sub(square(mu.ref));
	return x.sub(mu).div(sqrt(sigma2.add(opts.epsilon ?? 1e-5)));
}
/**
* One-hot encodes the given indices.
*
* Each index in the integer input `x` is encoded as a vector of zeros of length
* `numClasses`, with a 1 at the index position specified by its value.
*
* ```js
* import { nn, numpy as np } from '@jax-js/jax';
*
* nn.oneHot(np.array([1, 1, 2], { dtype: np.int32 }), 3);
* // Output:
* // [[0, 1, 0],
* //  [0, 1, 0],
* //  [0, 0, 1]]
* ```
*/
function oneHot(x, numClasses) {
	if (require_backend.isFloatDtype(x.dtype)) throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
	return eye(numClasses, void 0, { device: x.device }).slice(x);
}
/**
* Scaled dot product attention (SDPA).
*
* Computes `softmax((Q @ K^T) / sqrt(d) + bias) @ V`, where `Q` is the query,
* `K` is the key, `V` is the value, and `d` is the dimensionality of each key
* and query vector.
*
* Multi-query attention is applied when input `key` and `value` tensors have
* fewer heads than `query`.
*
* We use the following uppercase letters to denote array shapes:
* - `B` = batch size
* - `S` = length of key/value sequences (source)
* - `L` = length of query sequences
* - `N` = number of attention heads
* - `H` = dimensionality of each attention head
* - `K` = number of key/value heads (for grouped-query attention)
*
* The batch size `B` may be omitted, which is equivalent to `B = 1`. In this
* case it must be omitted from all inputs.
*
* @param query - Query array; shape `[B, L, N, H]`
* @param key - Key array; shape `[B, S, K, H]`
* @param value - Value array; same shape as `key`
* @param opts.bias - Optional bias to add to the attention logits; shape
*   `[B, N, L, S]` or broadcastable to it.
* @param opts.mask - Optional mask to apply to the attention logits; should be
*   a boolean array broadcastable to `[B, N, L, S]`, where `true` indicates
*   the element should take part in attention.
* @param opts.scale - Scaling factor override, default is `1 / sqrt(H)`.
* @param opts.isCausal - If true, applies a casual mask.
* @param opts.querySeqLengths - Optional sequence lengths for the queries;
*   shape `(B,)`. Taken from the beginning of the tensor.
* @param opts.keyValueSeqLengths - Optional sequence lengths for the keys and
*   values; shape `(B,)`. Taken from the beginning of the tensor.
* @param opts.localWindowSize - If specified, applies a local attention window
*   of the given size. Can be a single number or a tuple `[left, right]`.
*
* @returns The result of the attention operation; shape is the same as query
*   `[B, L, N, H]`, or `[L, N, H]` if `B` is omitted.
*/
function dotProductAttention(query, key$1, value, opts = {}) {
	query = fudgeArray(query);
	key$1 = fudgeArray(key$1);
	value = fudgeArray(value);
	if (query.ndim !== 3 && query.ndim !== 4 || query.ndim !== key$1.ndim || query.ndim !== value.ndim) throw new Error(`dotProductAttention: expected all tensors to have rank 3 or 4, got Q=${query.aval}, K=${key$1.aval}, V=${value.aval}`);
	if (!require_backend.deepEqual(key$1.shape, value.shape)) throw new Error(`dotProductAttention: key and value shapes must match, got K=${key$1.shape}, V=${value.shape}`);
	const isRank3 = query.ndim === 3;
	if (isRank3) {
		query = expandDims(query, 0);
		key$1 = expandDims(key$1, 0);
		value = expandDims(value, 0);
	}
	const [B, L, N, H] = query.shape;
	if (key$1.shape[0] !== B || key$1.shape[3] !== H) throw new Error(`dotProductAttention: query and key shapes mismatch, got Q=${query.aval}, K=${key$1.aval}`);
	const S = key$1.shape[1];
	const K = key$1.shape[2];
	if (N < K || N != K && N % K !== 0) throw new Error(`dotProductAttention: number of query heads N=${N} must be divisible by number of key/value heads K=${K} for GQA`);
	const G = N / K;
	key$1 = tile(key$1, [
		1,
		1,
		G,
		1
	]);
	value = tile(value, [
		1,
		1,
		G,
		1
	]);
	const scale = opts.scale ?? 1 / Math.sqrt(H);
	let scores = einsum("BLNH,BSNH->BNLS", query, key$1).mul(scale);
	if (opts.bias !== void 0) scores = scores.add(opts.bias);
	if (opts.mask !== void 0) scores = where(opts.mask, scores, -Infinity);
	if (opts.isCausal) {
		const causalMask = tri(L, S, 0, { dtype: require_backend.DType.Bool });
		scores = where(causalMask, scores, -Infinity);
	}
	if (opts.localWindowSize !== void 0) {
		const [before, after] = typeof opts.localWindowSize === "number" ? [opts.localWindowSize, opts.localWindowSize] : opts.localWindowSize;
		if (before < 0 || after < 0 || !Number.isInteger(before) || !Number.isInteger(after)) throw new Error(`dotProductAttention: localWindowSize values must be non-negative, got ${opts.localWindowSize}`);
		const localMask = tri(L, S, after, { dtype: require_backend.DType.Bool }).mul(tri(L, S, -before - 1, { dtype: require_backend.DType.Bool }).notEqual(true));
		scores = where(localMask, scores, -Infinity);
	}
	if (opts.querySeqLengths !== void 0) {
		const sl = expandDims(opts.querySeqLengths, [
			-1,
			-2,
			-3
		]);
		scores = where(arange(L).reshape([
			1,
			1,
			L,
			1
		]).less(sl), scores, -Infinity);
	}
	if (opts.keyValueSeqLengths !== void 0) {
		const sl = expandDims(opts.keyValueSeqLengths, [
			-1,
			-2,
			-3
		]);
		scores = where(arange(S).reshape([
			1,
			1,
			1,
			S
		]).less(sl), scores, -Infinity);
	}
	const attn = softmax(scores, -1);
	const out = einsum("BNLS,BSNH->BLNH", attn, value);
	return isRank3 ? out.reshape([
		L,
		N,
		H
	]) : out;
}

//#endregion
//#region src/library/random.ts
var random_exports = {};
__export(random_exports, {
	bernoulli: () => bernoulli,
	bits: () => bits,
	cauchy: () => cauchy,
	exponential: () => exponential,
	gumbel: () => gumbel,
	key: () => key,
	laplace: () => laplace,
	multivariateNormal: () => multivariateNormal,
	normal: () => normal,
	split: () => split,
	uniform: () => uniform
});
function validateKeyShape(key$1, scalar = false) {
	if (key$1.ndim === 0) throw new Error("Key must have at least one dimension.");
	if (key$1.shape[key$1.shape.length - 1] !== 2) throw new Error(`Invalid key shape: ${key$1.shape}. Expected last dimension to be 2.`);
	if (scalar && key$1.shape.length > 1) throw new Error(`Expected a single PRNG key, but got a batch of keys with shape ${JSON.stringify(key$1.shape)} - use jax.vmap for batching.`);
	return key$1.shape.slice(0, -1);
}
function getK01(key$1) {
	const keyShape = validateKeyShape(key$1, true);
	let [k0, k1] = split$2(key$1, -1, [1, 1]);
	k0 = k0.reshape(keyShape);
	k1 = k1.reshape(keyShape);
	return [k0, k1];
}
/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
function key(seed) {
	seed = array(seed, { dtype: require_backend.DType.Uint32 });
	if (seed.ndim !== 0) throw new Error(`key: seed must be a scalar integer, but got shape ${seed.shape} - use jax.vmap for batching.`);
	return stack([0, seed]);
}
/** Splits a PRNG key into `num` new keys by adding a leading axis. */
function split(key$1, num = 2) {
	const shape$1 = typeof num === "number" ? [num] : num;
	for (const len of shape$1) if (len <= 0 || !Number.isInteger(len)) throw new Error(`Invalid split length: ${len}. Must be a positive integer.`);
	const [k0, k1] = getK01(key$1);
	return stack([randomBits(k0.ref, k1.ref, shape$1, 0), randomBits(k0, k1, shape$1, 1)], -1);
}
/** Sample uniform bits in the form of unsigned integers. */
function bits(key$1, shape$1 = []) {
	const [k0, k1] = getK01(key$1);
	return randomBits(k0, k1, shape$1);
}
/**
* @function
* Sample uniform random values in [minval, maxval) with given shape.
*/
const uniform = jit$1(function uniform$1(key$1, shape$1 = [], { minval = 0, maxval = 1 } = {}) {
	if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
	const mantissa = bits(key$1, shape$1).div(array(512, {
		dtype: require_backend.DType.Uint32,
		device: key$1.device
	}));
	const float12 = mantissa.add(array(1065353216, {
		dtype: require_backend.DType.Uint32,
		device: key$1.device
	}));
	const rand = bitcast(float12, require_backend.DType.Float32).sub(1);
	if (minval === 0 && maxval === 1) return rand;
	else return rand.mul(maxval - minval).add(minval);
}, { staticArgnums: [1, 2] });
/**
* Sample Bernoulli random variables with given mean (0,1 categorical).
*
* Returns a random Boolean array with the specified shape. `p` can be an array
* and must be broadcastable to `shape`.
*/
function bernoulli(key$1, p = .5, shape$1 = []) {
	p = fudgeArray(p);
	return uniform(key$1, shape$1).less(p);
}
/**
* @function
* Sample from a Cauchy distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = tan(π * (u - 0.5))` where u ~ Uniform(0, 1).
*/
const cauchy = jit$1(function cauchy$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	return tan(u.sub(.5).mul(Math.PI));
}, { staticArgnums: [1] });
/**
* @function
* Sample exponential random values according to `p(x) = exp(-x)`.
*/
const exponential = jit$1(function exponential$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	return negative(log1p(negative(u)));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Gumbel distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = -log(-log(u))` where u ~ Uniform(0, 1).
*/
const gumbel = jit$1(function gumbel$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	return negative(log(negative(log1p(negative(u)))));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Laplace distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: the CDF is `F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))`.
* Inverting: `x = -sign(u - 0.5) * log(1 - 2 * |u - 0.5|)`.
*/
const laplace = jit$1(function laplace$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	const centered = u.sub(.5);
	const s = sign(centered.ref);
	const absVal = absolute(centered);
	return s.mul(log1p(absVal.mul(-2)).mul(-1));
}, { staticArgnums: [1] });
/**
* @function
* Sample multivariate normal random values with given mean and covariance.
*
* The values are returned with the given shape, along with the final dimension
* used to represent the n-dimensional multivariate normal factors.
*
* This uses Cholesky decomposition on the covariance matrix.
*
* - `key` - PRNG key
* - `mean` - Mean vector of shape `[..., n]`
* - `cov` - Covariance of shape `[..., n, n]`, must be positive-definite
* - `shape` - Result batch shape, must be broadcastable with
*            `mean.shape[:-1]` and `cov.shape[:-2]`
* @returns Random samples of shape `[...shape, n]`
*/
const multivariateNormal = jit$1(function multivariateNormal$1(key$1, mean$1, cov$1, shape$1 = []) {
	mean$1 = fudgeArray(mean$1);
	cov$1 = fudgeArray(cov$1);
	const n = mean$1.shape[mean$1.ndim - 1];
	if (cov$1.shape[cov$1.ndim - 1] !== n || cov$1.shape[cov$1.ndim - 2] !== n) throw new Error(`Invalid covariance shape: ${cov$1.shape}. Expected last two dimensions to be [${n}, ${n}].`);
	const outputShape = broadcastShapes(shape$1, mean$1.shape.slice(0, -1), cov$1.shape.slice(0, -2)).concat(n);
	const L = cholesky(cov$1);
	const z = normal(key$1, outputShape);
	return einsum("...ij,...j->...i", L, z).add(mean$1);
}, { staticArgnums: [3] });
/**
* @function
* Sample random values according to `p(x) = 1/sqrt(2pi) * exp(-x^2/2)`.
*
* Unlike JAX, this uses the Box-Muller transform. JAX uses the erf_inv primitive instead and
* directly inverts the CDF, but we don't have support for that yet. Outputs will not be
* bitwise identical to JAX.
*/
const normal = jit$1(function normal$1(key$1, shape$1 = []) {
	const [k1, k2] = split(key$1, 2);
	const u1 = uniform(k1, shape$1);
	const u2 = uniform(k2, shape$1);
	const radius = sqrt(log1p(negative(u1)).mul(-2));
	const theta = u2.mul(2 * Math.PI);
	return radius.mul(cos(theta));
}, { staticArgnums: [1] });

//#endregion
//#region src/library/scipy-special.ts
var scipy_special_exports = {};
__export(scipy_special_exports, {
	erf: () => erf,
	erfc: () => erfc,
	logSoftmax: () => logSoftmax,
	logit: () => logit,
	logsumexp: () => logsumexp,
	softmax: () => softmax
});
/**
* @function
* The logit function, `logit(p) = log(p / (1-p))`.
*/
const logit = jit$1(function logit$1(x) {
	return log(x.ref.div(subtract(1, x)));
});

//#endregion
//#region src/polyfills.ts
/** @file Polyfills for using this library. */
Symbol.dispose ??= Symbol.for("Symbol.dispose");
Symbol.asyncDispose ??= Symbol.for("Symbol.asyncDispose");

//#endregion
//#region src/index.ts
/**
* @function
* Compute the forward-mode Jacobian-vector product for a function.
*/
const jvp = jvp$1;
/**
* @function
* Vectorize an operation on a batched axis for one or more inputs.
*/
const vmap = vmap$1;
/**
* @function
* Compute the Jacobian evaluated column-by-column by forward-mode AD.
*/
const jacfwd = jacfwd$1;
/**
* @function
* Construct a Jaxpr by dynamically tracing a function with example inputs.
*/
const makeJaxpr = makeJaxpr$1;
/**
* @function
* Mark a function for automatic JIT compilation, with operator fusion.
*
* The function will be compiled the first time it is called with a set of
* argument shapes.
*
* You can call `.dispose()` on the returned, JIT-compiled function after all
* calls to free memory associated with array constants.
*
* **Options:**
* - `staticArgnums`: An array of argument indices to treat as static
*   (compile-time constant). These arguments must be hashable, won't be traced,
*   and different values will trigger recompilation.
* - `device`: The device to place the computation on. If not specified, the
*   computation will be placed on the default device.
*/
const jit = jit$1;
/**
* @function
* Produce a local linear approximation to a function at a point using jvp() and
* partial evaluation.
*/
const linearize = linearize$1;
/**
* @function
* Calculate the reverse-mode vector-Jacobian product for a function.
*
* The return value is a tuple of `[out, vjpFn]`, where `out` is the output of
* `f(primals)`, and `vjpFn` is a function that takes in cotangents for each
* output and returns the cotangents for each input.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return an
* `[out, aux]` tuple, and `vjp` returns `[out, vjpFn, aux]`.
*
* @example
* ```ts
* const [y, vjpFn] = vjp(f, [x]);
*
* // With hasAux
* const [y, vjpFn, aux] = vjp(f, [x], { hasAux: true });
* ```
*/
const vjp = vjp$1;
/**
* @function
* Compute the gradient of a scalar-valued function `f` with respect to its
* first argument.
*
* Pass in different `argnums` to differentiate with respect to other
* arguments. If a tuple is provided, the return value will be a tuple of
* gradients corresponding to each argument index.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return a
* `[out, aux]` tuple, and the return value will be `[gradient, aux]`.
*
* @example
* ```ts
* const gradient = grad(f)(x);
*
* // With `argnums`
* const [gradientX, gradientZ] = grad(f, { argnums: [0, 2] })(x, y, z);
*
* // With `hasAux`
* const [gradient, aux] = grad(f, { hasAux: true })(x);
* ```
*/
const grad = grad$1;
/**
* @function
* Create a function that evaluates both `f` and the gradient of `f`.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return an
* `[out, aux]` tuple, and the return value will be `[[out, aux], gradient]`.
*
* @example
* ```ts
* // Without hasAux
* const [value, gradient] = valueAndGrad(f)(x);
*
* // With hasAux
* const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);
* ```
*/
const valueAndGrad = valueAndGrad$1;
/**
* @function
* Compute the Jacobian evaluated row-by-row by reverse-mode AD.
*/
const jacrev = jacrev$1;
/**
* @function
* Compute the Hessian matrix of a scalar-valued function.
*
* The Hessian is the matrix of second-order partial derivatives of a function.
* This is implemented as `jacfwd(grad(f))`.
*
* @example
* ```ts
* const f = (x: np.Array) => np.sum(x.ref.mul(x.ref).mul(x)); // x^3
* const H = hessian(f)(np.array([1, 2, 3]));
* // H[i,j] = d^2f / dx_i dx_j
* ```
*/
const hessian = hessian$1;
/**
* Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
*
* This can be used to wait for the results of an intermediate computation to
* finish. It's recommended to call this regularly in an iterative computation
* to avoid queueing up too many pending operations.
*
* Does not consume reference to the arrays.
*/
async function blockUntilReady(x) {
	const promises = [];
	for (const leaf of leaves(x)) if (leaf instanceof Array$1) promises.push(leaf.blockUntilReady());
	await Promise.all(promises);
	return x;
}
/**
* Transfer `x` to `device`.
*
* `x` may be a nested container of arrays or scalars. The resulting structure
* is committed to the device.
*
* If `device` is not specified, this function behaves as identity if the input
* is already an `Array`, otherwise it places the scalar uncommitted on the
* default device.
*/
async function devicePut(x, device) {
	const [xflat, structure$1] = flatten(x);
	const yflat = await Promise.all(xflat.map((leaf) => {
		if (leaf instanceof Array$1) return device ? leaf._put(require_backend.getBackend(device)) : Promise.resolve(leaf);
		else return Promise.resolve(array(leaf, { device }));
	}));
	return unflatten(structure$1, yflat);
}

//#endregion
exports.Array = Array$1;
exports.ClosedJaxpr = ClosedJaxpr;
exports.DType = require_backend.DType;
exports.Jaxpr = Jaxpr;
exports.blockUntilReady = blockUntilReady;
exports.defaultDevice = require_backend.defaultDevice;
exports.devicePut = devicePut;
exports.devices = require_backend.devices;
exports.getBackend = require_backend.getBackend;
exports.grad = grad;
exports.hessian = hessian;
exports.init = require_backend.init;
exports.jacfwd = jacfwd;
exports.jacobian = jacrev;
exports.jacrev = jacrev;
exports.jit = jit;
exports.jvp = jvp;
Object.defineProperty(exports, 'lax', {
  enumerable: true,
  get: function () {
    return lax_exports;
  }
});
exports.linearize = linearize;
exports.makeJaxpr = makeJaxpr;
Object.defineProperty(exports, 'nn', {
  enumerable: true,
  get: function () {
    return nn_exports;
  }
});
Object.defineProperty(exports, 'numpy', {
  enumerable: true,
  get: function () {
    return numpy_exports;
  }
});
Object.defineProperty(exports, 'random', {
  enumerable: true,
  get: function () {
    return random_exports;
  }
});
Object.defineProperty(exports, 'scipySpecial', {
  enumerable: true,
  get: function () {
    return scipy_special_exports;
  }
});
exports.setDebug = require_backend.setDebug;
exports.setScanPathCallback = require_backend.setScanPathCallback;
Object.defineProperty(exports, 'tree', {
  enumerable: true,
  get: function () {
    return tree_exports;
  }
});
exports.valueAndGrad = valueAndGrad;
exports.vjp = vjp;
exports.vmap = vmap;