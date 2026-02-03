//#region src/pprint.ts
/** General class for pretty-printing expressions with indentation. */
var PPrint = class PPrint {
	constructor(indents, lines) {
		this.indents = indents;
		this.lines = lines;
	}
	/** Add a fixed amount of indentation to each line. */
	indent(spaces) {
		return new PPrint(this.indents.map((i) => i + spaces), this.lines);
	}
	/** Concatenate pretty-printed expressions with newlines. */
	concat(...items) {
		return new PPrint((this.indents ?? []).concat(...items.map((i) => i.indents)), (this.lines ?? []).concat(...items.map((i) => i.lines)));
	}
	/** Stack one block to the right of another one, sharing 1 common line. */
	stack(other) {
		if (!other.lines.length) return this;
		if (!this.lines.length) return other;
		const indent = this.indents[this.indents.length - 1];
		const s = this.lines[this.lines.length - 1];
		const indentedBlock = other.indent(indent + s.length);
		return new PPrint(this.indents.concat(indentedBlock.indents.slice(1)), this.lines.slice(0, -1).concat(s + " ".repeat(other.indents[0]) + other.lines[0], ...indentedBlock.lines.slice(1)));
	}
	/** Combine this block of lines into a formatted string. */
	toString() {
		return this.lines.map((line, i) => " ".repeat(this.indents[i]) + line).join("\n");
	}
	static pp(s) {
		const lines = s.toString().split("\n");
		return new PPrint(Array(lines.length).fill(0), lines);
	}
};

//#endregion
//#region src/utils.ts
/** @file Generic programming utilities with no dependencies on library code. */
let DEBUG = 0;
/**
* Set the debug level for verbose logging.
*
* 1. JIT compile logs
* 2. Shader code
* 3. Expressions and metadata
* 4. JIT programs, tuning details
* 5. Most verbose operation traces
*
* This is an experimental API and may change in behavior. Do not rely on this
* in production.
*/
function setDebug(level) {
	DEBUG = level;
}
let scanPathCallback = null;
/**
* Set a callback to be notified when scan implementations are chosen.
* Useful for testing to verify the expected code path is taken.
*
* @param callback - Function called with (path, backend, details) when scan path is chosen.
*                   Pass null to disable tracking.
*
* @example
* ```ts
* let usedPath: ScanPath | null = null;
* setScanPathCallback((path, backend) => { usedPath = path; });
* await jitScan(init, xs);
* expect(usedPath).toBe("fused");
* setScanPathCallback(null); // cleanup
* ```
*/
function setScanPathCallback(callback) {
	scanPathCallback = callback;
}
/** Internal: report scan path choice to registered callback. */
function reportScanPath(path, backend, details) {
	if (scanPathCallback) scanPathCallback(path, backend, details);
}
function assertNonNull(value) {}
function unzip2(pairs) {
	const lst1 = [];
	const lst2 = [];
	for (const [x, y] of pairs) {
		lst1.push(x);
		lst2.push(y);
	}
	return [lst1, lst2];
}
function zip(xs, ys) {
	return xs.map((x, i) => [x, ys[i]]);
}
function zipn(...arrays) {
	const minLength = Math.min(...arrays.map((x) => x.length));
	return Array.from({ length: minLength }, (_, i) => arrays.map((arr) => arr[i]));
}
function sorted(arr) {
	return [...arr].sort((a, b) => a - b);
}
function rep(length, value) {
	if (value instanceof Function) return new Array(length).fill(0).map((_, i) => value(i));
	return new Array(length).fill(value);
}
function prod(arr) {
	return arr.reduce((acc, x) => acc * x, 1);
}
function gcd(...values) {
	let a = 0;
	for (let b of values) while (b !== 0) [a, b] = [b, a % b];
	return Math.abs(a);
}
/** Shorthand for integer division, like in Python. */
function intdiv(a, b) {
	return Math.floor(a / b);
}
/** Clamp `x` to the range `[min, max]`. */
function clamp(x, min, max) {
	return Math.max(min, Math.min(max, x));
}
/** Check if two objects are deep equal. */
function deepEqual(a, b) {
	if (a === b) return true;
	if (typeof a !== "object" || typeof b !== "object") return false;
	if (a === null || b === null) return false;
	if (Object.keys(a).length !== Object.keys(b).length) return false;
	for (const key of Object.keys(a)) if (!deepEqual(a[key], b[key])) return false;
	return true;
}
/** Produces a union of maps of sets. This mutates `a`. */
function mapSetUnion(a, b) {
	if (!b) return a;
	for (const [key, setB] of b.entries()) {
		const setA = a.get(key);
		if (setA) for (const val of setB) setA.add(val);
		else a.set(key, setB);
	}
	return a;
}
/** Splits the list based on a condition, `false` first then `true`. */
function partitionList(which, array) {
	const falseList = [];
	const trueList = [];
	for (let i = 0; i < which.length; i++) if (which[i]) trueList.push(array[i]);
	else falseList.push(array[i]);
	return [falseList, trueList];
}
/** Compare two arrays of numbers lexicographically. */
function lexCompare(a, b) {
	const minLength = Math.min(a.length, b.length);
	for (let i = 0; i < minLength; i++) {
		if (a[i] < b[i]) return -1;
		if (a[i] > b[i]) return 1;
	}
	return a.length - b.length;
}
/** Check if an object is a number pair, i.e., a tuple of two numbers. */
function isNumberPair(x) {
	return Array.isArray(x) && x.length === 2 && typeof x[0] === "number" && typeof x[1] === "number";
}
/** Check an axis against number of dimensions, and resolve negative axes. */
function checkAxis(axis, ndim) {
	if (axis < -ndim || axis >= ndim) throw new Error(`Axis ${axis} out of bounds for array of dimension ${ndim}`);
	return axis < 0 ? axis + ndim : axis;
}
/** Normalize common axis argument for functions, defaulting to all axes. */
function normalizeAxis(axis, ndim) {
	if (axis === null) return range(ndim);
	else if (typeof axis === "number") return [checkAxis(axis, ndim)];
	else {
		const seen = /* @__PURE__ */ new Set();
		for (const a of axis) {
			const ca = checkAxis(a, ndim);
			if (seen.has(ca)) throw new Error(`Duplicate axis ${ca} passed to function`);
			seen.add(ca);
		}
		return sorted(seen);
	}
}
/** Check for an array of integers with no duplicates. */
function checkInts(indices) {
	if (typeof indices === "number") {
		if (!Number.isInteger(indices)) throw new TypeError(`Expected integer index, got ${indices}`);
	} else {
		const seen = /* @__PURE__ */ new Set();
		for (const i of indices) {
			if (!Number.isInteger(i)) throw new TypeError(`Expected integer indices, got ${i}`);
			if (seen.has(i)) throw new Error(`Duplicate index ${i} passed to function`);
			seen.add(i);
		}
	}
}
function range(start, stop, step = 1) {
	if (stop === void 0) {
		stop = start;
		start = 0;
	}
	const result = [];
	for (let i = start; i < stop; i += step) result.push(i);
	return result;
}
function isPermutation(axis, n) {
	if (axis.length !== n) return false;
	const seen = /* @__PURE__ */ new Set();
	for (const x of axis) {
		if (x < 0 || x >= n) return false;
		seen.add(x);
	}
	return seen.size === n;
}
function invertPermutation(axis) {
	const n = axis.length;
	if (!isPermutation(axis, n)) throw new Error("invertPermutation: axis is not a permutation");
	const result = new Array(n);
	for (let i = 0; i < n; i++) result[axis[i]] = i;
	return result;
}
/** Topologically sort a DAG, given terminal nodes and an ancestor function. */
function toposort(terminals, parents) {
	const childCounts = /* @__PURE__ */ new Map();
	const stack = [...new Set(terminals)];
	while (true) {
		const node = stack.pop();
		if (!node) break;
		for (const parent of parents(node)) if (childCounts.has(parent)) childCounts.set(parent, childCounts.get(parent) + 1);
		else {
			childCounts.set(parent, 1);
			stack.push(parent);
		}
	}
	for (const node of terminals) childCounts.set(node, childCounts.get(node) - 1);
	const order = [];
	const frontier = terminals.filter((n) => !childCounts.get(n));
	while (true) {
		const node = frontier.pop();
		if (!node) break;
		order.push(node);
		for (const parent of parents(node)) {
			const c = childCounts.get(parent) - 1;
			childCounts.set(parent, c);
			if (c == 0) frontier.push(parent);
		}
	}
	return order.reverse();
}
/**
* Returns the largest power of 2 less than or equal to `max`.
*
* If `hint` is nonzero, it will not return a number greater than the first
* power of 2 that is greater than or equal to `hint`.
*/
function findPow2(hint, max) {
	if (max < 1) throw new Error("max must be a positive integer");
	let ret = 1;
	while (ret < hint && 2 * ret <= max) ret *= 2;
	return ret;
}
/**
* Implements a NumPy-style generalized broadcast rule on two array shapes.
*
* "When operating on two arrays, NumPy compares their shapes element-wise. It
* starts with the trailing (i.e. rightmost) dimension and works its way left.
* Two dimensions are compatible when:
*   1. they are equal, or
*   2. one of them is 1."
*
* Throws a TypeError if the broadcast is not possible.
*
* <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
*/
function generalBroadcast(a, b) {
	const out = [];
	let i = a.length - 1;
	let j = b.length - 1;
	for (; i >= 0 && j >= 0; i--, j--) {
		const x = a[i];
		const y = b[j];
		if (x === y) out.push(x);
		else if (x === 1) out.push(y);
		else if (y === 1) out.push(x);
		else throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
	}
	for (; i >= 0; i--) out.push(a[i]);
	for (; j >= 0; j--) out.push(b[j]);
	return out.reverse();
}
function recursiveFlatten(ar) {
	if (!Array.isArray(ar)) return [ar];
	return ar.flat(Infinity);
}
/** Strip an outermost pair of nested parentheses from an expression, if any. */
function strip1(str) {
	if (str[0] === "(" && str[str.length - 1] === ")") return str.slice(1, -1);
	return str;
}
const _stagingbuf = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
/**
* Polynomial hashes modulo p are good at avoiding collisions in expectation.
* Probability-wise, it's good enough to be used for something like
* deduplicating seen compiler expressions, although it's not adversarial.
*
* See https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(number_theory)
*/
var FpHash = class FpHash {
	value = 8773157n;
	#update(x) {
		const base = 873192869n;
		const modulus = 3189051996290219n;
		this.value = (this.value * base + x) % modulus;
	}
	update(x) {
		if (typeof x === "string") {
			this.#update(BigInt(x.length));
			for (let i = 0; i < x.length; i++) this.#update(BigInt(199 + x.charCodeAt(i)));
		} else if (typeof x === "number") if (Number.isInteger(x)) this.#update(68265653n ^ BigInt(x));
		else {
			_stagingbuf.setFloat64(0, x, true);
			this.#update(_stagingbuf.getBigUint64(0, true));
		}
		else if (typeof x === "boolean") this.#update(x ? 69069841n : 63640693n);
		else if (typeof x === "bigint") this.#update(x ^ 71657401n);
		else if (x === null) this.#update(37832657n);
		else if (x === void 0) this.#update(18145117n);
		else x.hash(this);
		return this;
	}
	static hash(...values) {
		const h = new FpHash();
		for (const x of values) h.update(x);
		return h.value;
	}
};
/** Run a function while caching it inline inside a `Map`. */
function runWithCache(cache, key, thunk) {
	const keyStr = JSON.stringify(key);
	if (cache.has(keyStr)) return cache.get(keyStr);
	else {
		const value = thunk();
		cache.set(keyStr, value);
		return value;
	}
}

//#endregion
//#region src/alu.ts
/** A numerical data type for array contents. */
let DType = /* @__PURE__ */ function(DType$1) {
	DType$1["Float32"] = "float32";
	DType$1["Int32"] = "int32";
	DType$1["Uint32"] = "uint32";
	DType$1["Bool"] = "bool";
	DType$1["Float16"] = "float16";
	DType$1["Float64"] = "float64";
	return DType$1;
}({});
const byteWidth = (dtype) => {
	switch (dtype) {
		case DType.Float32:
		case DType.Int32:
		case DType.Uint32:
		case DType.Bool: return 4;
		case DType.Float16: return 2;
		case DType.Float64: return 8;
		default: throw new TypeError(`Unknown dtype: ${dtype}`);
	}
};
const isFloatDtype = (dtype) => dtype === DType.Float32 || dtype === DType.Float16 || dtype === DType.Float64;
/**
* Promote two dtypes to their join according to the type lattice.
*
* When performing operations between arrays of different types, we need to
* promote both operands to a common type that can represent values from both
* input types. This follows JAX's type promotion rules.
*
* **Type lattice:**
* ```text
* bool -> uint32 -> int32 -> float16 -> float32 -> float64
*  weakType --^
* ```
*
* `weakType` represents weakly typed arrays. These are created for JS numbers,
* which default to float32 but "weak" so they cast to the dtype of any array
* they are first combined with, except `bool`.
*
* **Examples:**
* - `promoteTypes(bool, int32) → int32`
* - `promoteTypes(uint32, int32) → int32`
* - `promoteTypes(int32, float16) → float16`
* - `promoteTypes(float16, float32) → float32`
* - `promoteTypes(uint32, float32) → float32`
*/
function promoteTypes(dtype1, dtype2) {
	if (dtype1 === dtype2) return dtype1;
	const rank = {
		[DType.Bool]: 0,
		[DType.Uint32]: 1,
		[DType.Int32]: 2,
		[DType.Float16]: 3,
		[DType.Float32]: 4,
		[DType.Float64]: 5
	};
	return rank[dtype1] > rank[dtype2] ? dtype1 : dtype2;
}
function dtypedArray(dtype, data) {
	const { buffer, byteLength, byteOffset } = data;
	const length = byteLength / byteWidth(dtype);
	switch (dtype) {
		case DType.Float32: return new Float32Array(buffer, byteOffset, length);
		case DType.Int32:
		case DType.Bool: return new Int32Array(buffer, byteOffset, length);
		case DType.Uint32: return new Uint32Array(buffer, byteOffset, length);
		case DType.Float16: return new Float16Array(buffer, byteOffset, length);
		case DType.Float64: return new Float64Array(buffer, byteOffset, length);
		default: throw new Error(`Unimplemented dtype: ${dtype}`);
	}
}
function dtypedJsArray(dtype, data) {
	switch (dtype) {
		case DType.Float32: return new Float32Array(data);
		case DType.Int32:
		case DType.Bool: return new Int32Array(data);
		case DType.Uint32: return new Uint32Array(data);
		case DType.Float16: return new Float16Array(data);
		case DType.Float64: return new Float64Array(data);
		default: throw new Error(`Unimplemented dtype: ${dtype}`);
	}
}
/**
* Mathematical expression on scalar values.
*
* This is similiar to and based on tinygrad's UOp class, but it's more specific
* to just math on scalars. We're doing this to avoid the complexity of a full
* graph rewrite engine.
*/
var AluExp = class AluExp {
	#hash;
	#simplified;
	#range;
	constructor(op, dtype, src, arg = void 0) {
		this.op = op;
		this.dtype = dtype;
		this.src = src;
		this.arg = arg;
		if (AluGroup.RequiredFloat.has(op) && !isFloatDtype(dtype)) throw new TypeError(`Unsupported dtype for ${op}: ${dtype}`);
		if (op === AluOp.Bitcast && (dtype === DType.Bool || src[0].dtype === DType.Bool || byteWidth(dtype) !== byteWidth(src[0].dtype))) throw new TypeError(`Bitcast from ${src[0].dtype} -> ${dtype}`);
		if (op === AluOp.Threefry2x32 && (dtype !== DType.Uint32 || src.some((x) => x.dtype !== DType.Uint32))) throw new TypeError("Threefry2x32 requires uint32 types");
	}
	static add(a, b) {
		return new AluExp(AluOp.Add, a.dtype, [a, b]);
	}
	static sub(a, b) {
		return new AluExp(AluOp.Sub, a.dtype, [a, b]);
	}
	static mul(a, b) {
		return new AluExp(AluOp.Mul, a.dtype, [a, b]);
	}
	static idiv(a, b) {
		return new AluExp(AluOp.Idiv, a.dtype, [a, b]);
	}
	static mod(a, b) {
		return new AluExp(AluOp.Mod, a.dtype, [a, b]);
	}
	static min(a, b) {
		return new AluExp(AluOp.Min, a.dtype, [a, b]);
	}
	static max(a, b) {
		return new AluExp(AluOp.Max, a.dtype, [a, b]);
	}
	static sin(a) {
		return new AluExp(AluOp.Sin, a.dtype, [a]);
	}
	static cos(a) {
		return new AluExp(AluOp.Cos, a.dtype, [a]);
	}
	static asin(a) {
		return new AluExp(AluOp.Asin, a.dtype, [a]);
	}
	static atan(a) {
		return new AluExp(AluOp.Atan, a.dtype, [a]);
	}
	static exp(a) {
		return new AluExp(AluOp.Exp, a.dtype, [a]);
	}
	static log(a) {
		return new AluExp(AluOp.Log, a.dtype, [a]);
	}
	static erf(a) {
		return new AluExp(AluOp.Erf, a.dtype, [a]);
	}
	static erfc(a) {
		return new AluExp(AluOp.Erfc, a.dtype, [a]);
	}
	static sqrt(a) {
		return new AluExp(AluOp.Sqrt, a.dtype, [a]);
	}
	static floor(a) {
		if (!isFloatDtype(a.dtype)) return a;
		return new AluExp(AluOp.Floor, a.dtype, [a]);
	}
	static ceil(a) {
		if (!isFloatDtype(a.dtype)) return a;
		return new AluExp(AluOp.Ceil, a.dtype, [a]);
	}
	static reciprocal(a) {
		return new AluExp(AluOp.Reciprocal, a.dtype, [a]);
	}
	static cast(dtype, a) {
		if (a.dtype === dtype) return a;
		return new AluExp(AluOp.Cast, dtype, [a]);
	}
	static bitcast(dtype, a) {
		if (a.dtype === dtype) return a;
		return new AluExp(AluOp.Bitcast, dtype, [a]);
	}
	static threefry2x32(k0, k1, c0, c1, mode = "xor") {
		return new AluExp(AluOp.Threefry2x32, DType.Uint32, [
			k0,
			k1,
			c0,
			c1
		], mode);
	}
	static cmplt(a, b) {
		return new AluExp(AluOp.Cmplt, DType.Bool, [a, b]);
	}
	static cmpne(a, b) {
		return new AluExp(AluOp.Cmpne, DType.Bool, [a, b]);
	}
	static where(cond, a, b) {
		return new AluExp(AluOp.Where, a.dtype, [
			cond,
			a,
			b
		]);
	}
	static const(dtype, value) {
		if (dtype === DType.Bool) value = Number(Boolean(value));
		else if (dtype === DType.Int32) value = Math.trunc(value) | 0;
		else if (dtype === DType.Uint32) value = Math.trunc(value) >>> 0;
		if (typeof value !== "number") throw new TypeError(`Expected a number for constant, got ${typeof value}: ${value}`);
		return new AluExp(AluOp.Const, dtype, [], value);
	}
	static special(dtype, name, n) {
		return new AluExp(AluOp.Special, dtype, [], [name, n]);
	}
	static variable(dtype, name) {
		return new AluExp(AluOp.Variable, dtype, [], name);
	}
	static globalIndex(dtype, gid, len, bufidx) {
		return new AluExp(AluOp.GlobalIndex, dtype, [bufidx], [gid, len]);
	}
	static globalView(dtype, gid, st, indices) {
		return new AluExp(AluOp.GlobalView, dtype, indices, [gid, st]);
	}
	static f32(value) {
		return AluExp.const(DType.Float32, value);
	}
	static i32(value) {
		return AluExp.const(DType.Int32, value);
	}
	static u32(value) {
		return AluExp.const(DType.Uint32, value);
	}
	static bool(value) {
		return AluExp.const(DType.Bool, Number(value));
	}
	static f16(value) {
		return AluExp.const(DType.Float16, value);
	}
	static f64(value) {
		return AluExp.const(DType.Float64, value);
	}
	not() {
		if (this.dtype !== DType.Bool) throw new Error("not() can only be called on boolean expressions");
		return AluExp.cmpne(this, AluExp.const(DType.Bool, true));
	}
	/** Compute a reasonable expression hash with low collision rate. */
	getHash() {
		if (this.#hash !== void 0) return this.#hash;
		const hasher = new FpHash();
		hasher.update(this.op);
		hasher.update(this.dtype);
		if (this.op === AluOp.Const) hasher.update(this.arg);
		else hasher.update(JSON.stringify(this.arg));
		hasher.update(this.src.length);
		for (const s of this.src) hasher.update(s);
		this.#hash = hasher.value;
		return this.#hash;
	}
	hash(state) {
		state.update(this.getHash());
	}
	/** Substitute variables in this AluExp to values. */
	substitute(variables) {
		return this.rewrite((exp) => {
			if (exp.op === AluOp.Variable && Object.hasOwn(variables, exp.arg)) {
				if (exp.dtype !== variables[exp.arg].dtype) throw new Error(`Type mismatch: ${exp.dtype} vs ${variables[exp.arg].dtype}`);
				return variables[exp.arg];
			}
		});
	}
	/** Reindex gid values in this expression as needed. */
	reindexGids(newGids) {
		return this.rewrite((exp) => {
			if (exp.op === AluOp.GlobalIndex) {
				const [gid, len] = exp.arg;
				const newGid = newGids[gid];
				if (newGid !== gid) return AluExp.globalIndex(exp.dtype, newGid, len, exp.src[0]);
			} else if (exp.op === AluOp.GlobalView) {
				const gid = exp.arg[0];
				const newGid = newGids[gid];
				if (newGid !== gid) return AluExp.globalView(exp.dtype, newGid, exp.arg[1], exp.src);
			}
		});
	}
	#computeRange() {
		if (this.#range !== void 0) return this.#range;
		const src = this.src;
		const minMax4 = (f) => {
			const [r1, r2] = [src[0].#computeRange(), src[1].#computeRange()];
			const values = [
				f(r1[0], r2[0]),
				f(r1[0], r2[1]),
				f(r1[1], r2[0]),
				f(r1[1], r2[1])
			];
			return [Math.min(...values), Math.max(...values)];
		};
		let ret;
		switch (this.op) {
			case AluOp.Add:
				ret = [src[0].min + src[1].min, src[0].max + src[1].max];
				break;
			case AluOp.Sub:
				ret = [src[0].min - src[1].max, src[0].max - src[1].min];
				break;
			case AluOp.Mul:
				ret = minMax4((a, b) => a * b);
				break;
			case AluOp.Idiv:
				ret = minMax4((a, b) => Math.trunc(a / b));
				break;
			case AluOp.Mod: {
				let divisorRange = src[1].#computeRange();
				if (divisorRange[0] <= 0 && divisorRange[1] >= 0) divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
				if (divisorRange[1] < 0) divisorRange = [-divisorRange[1], -divisorRange[0]];
				const maxDivisor = isFloatDtype(this.dtype) ? divisorRange[1] : divisorRange[1] - 1;
				ret = [clamp(src[0].min, -maxDivisor, 0), clamp(src[0].max, 0, maxDivisor)];
				break;
			}
			case AluOp.Min:
				ret = [Math.min(src[0].min, src[1].min), Math.min(src[0].max, src[1].max)];
				break;
			case AluOp.Max:
				ret = [Math.max(src[0].min, src[1].min), Math.max(src[0].max, src[1].max)];
				break;
			case AluOp.Sin:
				ret = [-1, 1];
				break;
			case AluOp.Cos:
				ret = [-1, 1];
				break;
			case AluOp.Asin:
				ret = [-Math.PI / 2, Math.PI / 2];
				break;
			case AluOp.Atan:
				ret = [-Math.PI / 2, Math.PI / 2];
				break;
			case AluOp.Exp:
				ret = [Math.exp(src[0].min), Math.exp(src[0].max)];
				break;
			case AluOp.Log:
				ret = [Math.log(src[0].min), Math.log(src[0].max)];
				break;
			case AluOp.Erf:
				ret = [erf(src[0].min), erf(src[0].max)];
				break;
			case AluOp.Erfc:
				ret = [erfc(src[0].max), erfc(src[0].min)];
				break;
			case AluOp.Sqrt:
				ret = [Math.sqrt(src[0].min), Math.sqrt(src[0].max)];
				break;
			case AluOp.Floor:
				ret = [Math.floor(src[0].min), Math.floor(src[0].max)];
				break;
			case AluOp.Ceil:
				ret = [Math.ceil(src[0].min), Math.ceil(src[0].max)];
				break;
			case AluOp.Reciprocal:
				if (src[0].min <= 0 && src[0].max >= 0) return [-Infinity, Infinity];
				ret = [1 / src[0].max, 1 / src[0].min];
				break;
			case AluOp.Cast: {
				const wasFloat = isFloatDtype(src[0].dtype);
				const bounded = Number.isFinite(src[0].min) && Number.isFinite(src[0].max);
				if (this.dtype === DType.Bool) {
					const canBeZero = src[0].min <= 0 && src[0].max >= 0;
					const mustBeZero = src[0].min === 0 && src[0].max === 0;
					ret = mustBeZero ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
				} else if (this.dtype === DType.Int32) {
					const a = wasFloat ? clamp(src[0].min, -2147483648, 2147483647) | 0 : src[0].min | 0;
					const b = wasFloat ? clamp(src[0].max, -2147483648, 2147483647) | 0 : src[0].max | 0;
					ret = bounded && a <= b ? [a, b] : [-Infinity, Infinity];
				} else if (this.dtype === DType.Uint32) {
					const a = wasFloat ? clamp(src[0].min, 0, 4294967295) >>> 0 : src[0].min >>> 0;
					const b = wasFloat ? clamp(src[0].max, 0, 4294967295) >>> 0 : src[0].max >>> 0;
					ret = bounded && a <= b ? [a, b] : [0, Infinity];
				} else ret = [src[0].min, src[0].max];
				break;
			}
			case AluOp.Cmplt:
				ret = [0, 1];
				break;
			case AluOp.Cmpne:
				ret = [0, 1];
				break;
			case AluOp.Where:
				ret = [Math.min(src[1].min, src[2].min), Math.max(src[1].max, src[2].max)];
				break;
			case AluOp.Const:
				ret = [this.arg, this.arg];
				break;
			case AluOp.Special:
				ret = [0, this.arg[1] - 1];
				break;
			default: ret = [-Infinity, Infinity];
		}
		if (isNaN(ret[0]) || isNaN(ret[1])) ret = [-Infinity, Infinity];
		if (this.dtype === DType.Bool) {
			ret[0] = clamp(ret[0], 0, 1);
			ret[1] = clamp(ret[1], 0, 1);
		}
		if (this.dtype === DType.Uint32) ret[0] = Math.max(0, ret[0]);
		this.#range = ret;
		return ret;
	}
	get min() {
		return this.#computeRange()[0];
	}
	get max() {
		return this.#computeRange()[1];
	}
	/** Largest known integer that divides self. */
	constFactor() {
		if (this.op === AluOp.Const) return Math.abs(this.arg);
		if (this.op === AluOp.Add) return gcd(this.src[0].constFactor(), this.src[1].constFactor());
		if (this.op === AluOp.Mul) {
			if (this.src[0].op === AluOp.Const) return Math.abs(this.src[0].arg);
			if (this.src[1].op === AluOp.Const) return Math.abs(this.src[1].arg);
		}
		return 1;
	}
	/**
	* Checks if divisible by an integer v and returns the quotient if it is, or
	* `null` if it's not divisible.
	*/
	divides(v) {
		if (v === 1) return this;
		if (this.op === AluOp.Const && this.arg % v === 0) return AluExp.const(this.dtype, this.arg / v);
		if (this.op === AluOp.Add) {
			const a = this.src[0].divides(v);
			if (a !== null) {
				const b = this.src[1].divides(v);
				if (b !== null) return AluExp.add(a, b);
			}
		}
		if (this.op === AluOp.Mul) {
			const a = this.src[0].divides(v);
			if (a !== null) return AluExp.mul(a, this.src[1]);
			const b = this.src[1].divides(v);
			if (b !== null) return AluExp.mul(this.src[0], b);
		}
		return null;
	}
	#isConstInt() {
		return this.op === AluOp.Const && (this.dtype === DType.Int32 || this.dtype === DType.Uint32);
	}
	/**
	* Get all expressions by deeply matching an operation.
	*
	* For example: `((2+(3*5))+4).splitOp(+) -> [2,(3*5),4]`.
	*/
	*splitOp(sep) {
		if (this.op === sep) for (const src of this.src) yield* src.splitOp(sep);
		else yield this;
	}
	/**
	* Simplify the expression by replacing any known patterns and deduping
	* identical subexpressions.
	*/
	simplify(cache = /* @__PURE__ */ new Map()) {
		if (this.#simplified !== void 0) return this.#simplified;
		const hash = this.getHash();
		const prevCachedValue = cache.get(hash);
		if (prevCachedValue !== void 0) return this.#simplified = prevCachedValue;
		const simplified = this.#simplifyInner(cache);
		const simplifiedHash = simplified.getHash();
		const prevSimplified = cache.get(simplifiedHash);
		if (prevSimplified !== void 0) {
			cache.set(hash, prevSimplified);
			this.#simplified = prevSimplified;
			return prevSimplified;
		} else {
			cache.set(hash, simplified);
			cache.set(simplifiedHash, simplified);
			this.#simplified = simplified;
			return simplified;
		}
	}
	#simplifyInner(cache) {
		const src = this.src.map((x) => x.simplify(cache));
		const { op } = this;
		if (src.every((x) => x.op === AluOp.Const) && !AluGroup.Variable.has(op)) {
			const newExp$1 = new AluExp(op, this.dtype, src, this.arg);
			return AluExp.const(this.dtype, newExp$1.evaluate({}));
		}
		if (op !== AluOp.Const && this.min === this.max) return AluExp.const(this.dtype, this.min);
		if (AluGroup.Binary.has(op)) for (let i = 0; i < 2; i++) {
			if (src[i].op !== AluOp.Const) continue;
			const x = src[i].arg;
			if (op === AluOp.Add && x === 0) return src[1 - i];
			if (op === AluOp.Sub && i === 1 && x === 0) return src[1 - i];
			if (op === AluOp.Mul && x === 1) return src[1 - i];
			if (op === AluOp.Mul && x === 0) return AluExp.const(this.dtype, 0);
			if (op === AluOp.Idiv && i === 1 && x === 1 && !isFloatDtype(this.dtype)) return src[1 - i];
			if (op === AluOp.Cmpne && src[i].dtype === DType.Bool && x === 0) return src[1 - i];
		}
		if ((op === AluOp.Add || op === AluOp.Sub) && src[1].op === AluOp.Mul) {
			const [a, b] = src[1].src;
			const opNeg = op === AluOp.Add ? AluOp.Sub : AluOp.Add;
			if (a.op === AluOp.Const && a.arg === -1) return new AluExp(opNeg, this.dtype, [src[0], b]);
			else if (b.op === AluOp.Const && b.arg === -1) return new AluExp(opNeg, this.dtype, [src[0], a]);
		}
		if (op === AluOp.Where && src.slice(1).every((s, i) => s.op === AluOp.Const && s.arg === 1 - i)) return AluExp.cast(this.dtype, src[0]);
		if (op === AluOp.Cmplt) {
			if (src[0].min >= src[1].max) return AluExp.const(DType.Bool, false);
			if (src[0].max < src[1].min) return AluExp.const(DType.Bool, true);
		}
		if (op === AluOp.Cmpne) {
			if (src[0].max < src[1].min || src[0].min > src[1].max) return AluExp.const(DType.Bool, true);
		}
		if (op === AluOp.Where) {
			if (src[0].max === 0) return src[2];
			if (src[0].min === 1) return src[1];
		}
		if (op === AluOp.Mod && src[1].op === AluOp.Const && src[0].min >= 0 && src[0].max < src[1].arg) return src[0];
		if (op === AluOp.Mod && src[0].op === AluOp.Mod && src[1].#isConstInt() && src[0].src[1].#isConstInt()) {
			const A = src[0].src[1].arg;
			const B = src[1].arg;
			if (A > 0 && B > 0 && (A % B === 0 || B % A === 0)) return AluExp.mod(src[0].src[0], AluExp.const(this.dtype, Math.min(A, B))).simplify();
		}
		if (op === AluOp.Add && src[0].op === AluOp.Mul && src[0].src[1].#isConstInt() && src[1].op === AluOp.Mod && src[1].src[1].#isConstInt() && src[0].src[1].arg === src[1].src[1].arg) {
			const [mul, mod] = src;
			const check = (exp) => {
				return exp.op === AluOp.Idiv && exp.src[1].#isConstInt() && exp.src[1].arg === mod.src[1].arg && exp.src[0] === mod.src[0];
			};
			if (check(mul.src[0])) return mod.src[0];
			if (mul.src[0].op === AluOp.Mod) {
				const [x, y] = mul.src[0].src;
				if (check(x)) return AluExp.mod(mod.src[0], AluExp.mul(mod.src[1], y)).simplify(cache);
			}
		}
		if (op === AluOp.Idiv && src[1].#isConstInt()) {
			const [numer, denom] = src;
			const B = denom.arg;
			for (let i = 0; i < 2; i++) {
				if (numer.op === AluOp.Mul && numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					if (A % B === 0) {
						let ret = numer.src[1 - i];
						if (A / B !== 1) ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
						return ret.simplify(cache);
					}
				}
				for (let j = 0; j < 2; j++) if (numer.op === AluOp.Add && numer.src[j].op === AluOp.Mul && numer.src[j].src[i].#isConstInt()) {
					const A = numer.src[j].src[i].arg;
					if (A % B === 0) {
						let ret = numer.src[j].src[1 - i];
						if (A / B !== 1) ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
						ret = AluExp.add(ret, AluExp.idiv(numer.src[1 - j], AluExp.const(ret.dtype, B)));
						return ret.simplify(cache);
					}
				}
			}
		}
		if (op === AluOp.Mod && src[1].#isConstInt() && src[1].arg > 0 && src[0].min >= 0) {
			const [numer, denom] = src;
			const B = denom.arg;
			for (let i = 0; i < 2; i++) if (numer.op === AluOp.Add) {
				if (numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					const x = numer.src[1 - i];
					if (A % B === 0 && x.min >= 0) return AluExp.mod(x, denom).simplify(cache);
				}
				for (let j = 0; j < 2; j++) if (numer.src[i].op === AluOp.Mul && numer.src[i].src[j].#isConstInt()) {
					const A = numer.src[i].src[j].arg;
					const x = numer.src[1 - i];
					if (A % B === 0 && x.min >= 0) return AluExp.mod(x, denom).simplify(cache);
				}
			} else if (numer.op === AluOp.Mul) {
				if (numer.src[i].#isConstInt()) {
					const A = numer.src[i].arg;
					if (A % B === 0) return AluExp.const(this.dtype, 0);
					if (A % B === 1) return AluExp.mod(numer.src[1 - i], denom).simplify(cache);
				}
			}
		}
		const commOps = [
			AluOp.Add,
			AluOp.Mul,
			AluOp.Max,
			AluOp.Min
		];
		if (commOps.includes(op)) {
			const p = (a, b) => new AluExp(op, this.dtype, [a, b]);
			if (src[0].op === AluOp.Const) return p(src[1], src[0]).simplify(cache);
			if (src[0].op === op && src[0].src[1].op === AluOp.Const) if (src[1].op === AluOp.Const) return p(src[0].src[0], p(src[0].src[1], src[1])).simplify(cache);
			else return p(p(src[0].src[0], src[1]), src[0].src[1]).simplify(cache);
			if (src[1].op === op && src[1].src[1].op === AluOp.Const) return p(p(src[0], src[1].src[0]), src[1].src[1]).simplify(cache);
		}
		if ((op === AluOp.Mod || op === AluOp.Idiv) && src[1].#isConstInt()) {
			const [x, y] = src;
			{
				const factors = [];
				const terms = [];
				for (const u of x.splitOp(AluOp.Add)) {
					const factor = u.constFactor();
					factors.push(factor);
					terms.push(u.divides(factor));
				}
				const g = gcd(y.arg, ...factors);
				if (g !== 1) {
					let ret = new AluExp(op, this.dtype, [factors.map((f, i) => AluExp.mul(AluExp.const(terms[i].dtype, f / g), terms[i])).reduceRight((a, x$1) => AluExp.add(x$1, a)), AluExp.const(y.dtype, y.arg / g)]);
					if (op === AluOp.Mod) ret = AluExp.mul(ret, AluExp.const(this.dtype, g));
					return ret.simplify(cache);
				}
			}
			if (y.arg > 0 && x.min >= 0) {
				let [xNoConst, constVal] = [x, 0];
				if (x.op === AluOp.Add && x.src[1].op === AluOp.Const) [xNoConst, constVal] = [x.src[0], x.src[1].arg];
				const terms = [];
				const factors = [];
				for (const u of xNoConst.splitOp(AluOp.Add)) {
					const f = u.constFactor();
					const divided = u.divides(f);
					terms.push(divided ?? u);
					factors.push(divided ? f : 1);
				}
				const quotients = factors.map((f) => Math.floor(f / y.arg));
				const remainders = factors.map((f) => f % y.arg);
				const gcdVal = remainders.reduce((g, r) => gcd(g, r), y.arg);
				if (constVal % y.arg !== constVal || gcdVal !== 1 || remainders.some((r, i) => r === 0 || r !== factors[i] && op === AluOp.Mod)) {
					let quo = AluExp.const(x.dtype, Math.floor(constVal / y.arg));
					let rem = AluExp.const(x.dtype, Math.floor(constVal % y.arg / gcdVal));
					for (let i = 0; i < terms.length; i++) if (op === AluOp.Idiv && remainders[i] !== 0) rem = AluExp.add(rem, AluExp.mul(AluExp.const(x.dtype, Math.floor(factors[i] / gcdVal)), terms[i]));
					else {
						rem = AluExp.add(rem, AluExp.mul(AluExp.const(x.dtype, Math.floor(remainders[i] / gcdVal)), terms[i]));
						quo = AluExp.add(quo, AluExp.mul(AluExp.const(x.dtype, quotients[i]), terms[i]));
					}
					if (rem.min >= 0) if (op === AluOp.Mod) return AluExp.add(AluExp.mul(AluExp.const(x.dtype, gcdVal), AluExp.mod(rem, AluExp.const(x.dtype, Math.floor(y.arg / gcdVal)))), AluExp.const(x.dtype, constVal % gcdVal)).simplify(cache);
					else return AluExp.add(AluExp.idiv(rem, AluExp.const(x.dtype, Math.floor(y.arg / gcdVal))), quo).simplify(cache);
				}
			}
		}
		const newExp = src.every((s, i) => s === this.src[i]) ? this : new AluExp(op, this.dtype, src, this.arg);
		return newExp;
	}
	/** Resolve this to a value, or `undefined` if not possible. */
	resolve() {
		const x = this.simplify();
		if (x.op === AluOp.Const) return x.arg;
		return void 0;
	}
	/**
	* Evaluate the expression on CPU, returning the result.
	*
	* Typically you would compile the AluExp as a representation to a lower-level
	* language. This is just to define the semantics and help debug.
	*
	* Note that the representation of Bool is as a number (0 or 1) here.
	*/
	evaluate(context, globals) {
		if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
			const x = this.src[0].evaluate(context, globals);
			const y = this.src[1].evaluate(context, globals);
			switch (this.op) {
				case AluOp.Add: return this.dtype === DType.Bool ? Number(x || y) : x + y;
				case AluOp.Sub: return x - y;
				case AluOp.Mul: return this.dtype === DType.Bool ? Number(x && y) : x * y;
				case AluOp.Idiv: return Math.trunc(x / y);
				case AluOp.Mod: return x % y;
				case AluOp.Min: return Math.min(x, y);
				case AluOp.Max: return Math.max(x, y);
				case AluOp.Cmplt: return Number(x < y);
				case AluOp.Cmpne: return Number(x != y);
				default: throw new Error(`Missing implemementation for ${this.op}`);
			}
		}
		if (AluGroup.Unary.has(this.op)) {
			const x = this.src[0].evaluate(context, globals);
			switch (this.op) {
				case AluOp.Sin: return Math.sin(x);
				case AluOp.Cos: return Math.cos(x);
				case AluOp.Asin: return Math.asin(x);
				case AluOp.Atan: return Math.atan(x);
				case AluOp.Exp: return Math.exp(x);
				case AluOp.Log: return Math.log(x);
				case AluOp.Erf: return erf(x);
				case AluOp.Erfc: return erfc(x);
				case AluOp.Sqrt: return Math.sqrt(x);
				case AluOp.Floor: return Math.floor(x);
				case AluOp.Ceil: return Math.ceil(x);
				case AluOp.Reciprocal: return 1 / x;
				case AluOp.Cast: {
					const wasFloat = isFloatDtype(this.src[0].dtype);
					if (this.dtype === DType.Int32) return (wasFloat ? clamp(x, -2147483648, 2147483647) : x) | 0;
					else if (this.dtype === DType.Uint32) return (wasFloat ? clamp(x, 0, 4294967295) : x) >>> 0;
					else if (isFloatDtype(this.dtype)) return x;
					else if (this.dtype === DType.Bool) return Number(Boolean(x));
					else throw new Error(`Unsupported cast to ${this.dtype}`);
				}
				case AluOp.Bitcast: {
					const buf = new ArrayBuffer(byteWidth(this.dtype));
					const view = new DataView(buf);
					const fromType = this.src[0].dtype;
					if (fromType === DType.Float32) view.setFloat32(0, x, true);
					else if (fromType === DType.Int32) view.setInt32(0, x, true);
					else if (fromType === DType.Uint32) view.setUint32(0, x, true);
					else if (fromType === DType.Float16) view.setFloat16(0, x, true);
					else if (fromType === DType.Float64) view.setFloat64(0, x, true);
					else throw new Error(`Unsupported bitcast from ${fromType}`);
					if (this.dtype === DType.Float32) return view.getFloat32(0, true);
					else if (this.dtype === DType.Int32) return view.getInt32(0, true);
					else if (this.dtype === DType.Uint32) return view.getUint32(0, true);
					else if (this.dtype === DType.Float16) return view.getFloat16(0, true);
					else if (this.dtype === DType.Float64) return view.getFloat64(0, true);
					else throw new Error(`Unsupported bitcast to ${this.dtype}`);
				}
				default: throw new Error(`Missing implemementation for ${this.op}`);
			}
		}
		switch (this.op) {
			case AluOp.Where: return this.src[0].evaluate(context, globals) ? this.src[1].evaluate(context, globals) : this.src[2].evaluate(context, globals);
			case AluOp.Threefry2x32: {
				const [k0, k1, c0, c1] = this.src.map((x) => x.evaluate(context, globals));
				const [x0, x1] = threefry2x32(k0, k1, c0, c1);
				if (this.arg === "xor") return (x0 ^ x1) >>> 0;
				else if (this.arg === 0) return x0;
				else if (this.arg === 1) return x1;
				else throw new Error(`Invalid Threefry2x32 mode: ${this.arg}`);
			}
			case AluOp.Const: return this.arg;
			case AluOp.Special: {
				const x = context[this.arg[0]];
				if (x === void 0) throw new Error(`Missing special: ${this.arg[0]}`);
				return x;
			}
			case AluOp.Variable: {
				const x = context[this.arg];
				if (x === void 0) throw new Error(`Missing variable: ${this.arg}`);
				return x;
			}
			case AluOp.GlobalIndex: {
				if (!globals) throw new Error("Missing globals function");
				const gid = this.arg[0];
				const bufidx = this.src[0].evaluate(context, globals);
				return globals(gid, bufidx);
			}
			case AluOp.GlobalView: {
				if (!globals) throw new Error("Missing globals function");
				const gid = this.arg[0];
				const st = this.arg[1];
				const [iexpr, vexpr] = st.toAluExp(this.src);
				if (vexpr.evaluate(context, globals)) {
					const bufidx = iexpr.evaluate(context, globals);
					return globals(gid, bufidx);
				} else return 0;
			}
			default: throw new Error(`Missing implemementation for ${this.op}`);
		}
	}
	/** Get this expression in debug format as a string. */
	toString() {
		const BIN_SYM = {
			[AluOp.Add]: "+",
			[AluOp.Sub]: "-",
			[AluOp.Mul]: "*",
			[AluOp.Idiv]: "/",
			[AluOp.Mod]: "%"
		};
		const CMP_SYM = {
			[AluOp.Cmplt]: "<",
			[AluOp.Cmpne]: "!="
		};
		const UNARY_SYM = { [AluOp.Reciprocal]: "1/" };
		return this.fold((node, parts) => {
			switch (node.op) {
				case AluOp.Const: return "" + (node.dtype === DType.Bool ? Boolean(node.arg) : node.arg);
				case AluOp.Variable: return `$${node.arg}:${node.dtype}`;
				case AluOp.Special: {
					const [name, n] = node.arg;
					return `#${name}{${n}}`;
				}
				case AluOp.GlobalIndex: return `G_${node.arg[0]}<${node.dtype}>[${strip1(parts[0])}]`;
				case AluOp.GlobalView: {
					const [gid, st] = node.arg;
					const shape = st.shape.join(",");
					const lastStrides = st.lastStrides.join(",");
					const cont = st.contiguous ? "c" : "nc";
					return `GV_${gid}<${node.dtype}>{${shape}:${lastStrides}:${cont}}[${parts.map(strip1).join(", ")}]`;
				}
			}
			if (BIN_SYM[node.op]) return `(${parts[0]} ${BIN_SYM[node.op]} ${parts[1]})`;
			if (CMP_SYM[node.op]) return `(${parts[0]} ${CMP_SYM[node.op]} ${parts[1]})`;
			if (UNARY_SYM[node.op]) return `${UNARY_SYM[node.op]}${parts[0]}`;
			if (node.op === AluOp.Cast) return `Cast<${node.dtype}>(${strip1(parts[0])})`;
			if (node.op === AluOp.Bitcast) return `Bitcast<${node.dtype}>(${strip1(parts[0])})`;
			return `${node.op}(${parts.map(strip1).join(", ")})`;
		});
	}
	/** Generic fold() operation with a reducer over the expression tree. */
	fold(reducer) {
		const visited = /* @__PURE__ */ new Map();
		const recurse = (exp) => {
			if (visited.has(exp)) return visited.get(exp);
			const mappedSrc = exp.src.map((s) => recurse(s));
			const result = reducer(exp, mappedSrc);
			visited.set(exp, result);
			return result;
		};
		return recurse(this);
	}
	/** Check if any expression in the tree satisfies a predicate. */
	some(predicate) {
		const visited = /* @__PURE__ */ new Set();
		const recurse = (exp) => {
			if (visited.has(exp)) return false;
			if (predicate(exp)) return true;
			visited.add(exp);
			return exp.src.some(recurse);
		};
		return recurse(this);
	}
	/** Rewrite the expression recursively using a visitor. */
	rewrite(visitor) {
		return this.fold((exp, newSrc) => {
			if (newSrc.length === exp.src.length && newSrc.every((s, i) => s === exp.src[i])) return visitor(exp) ?? exp;
			else {
				const newExp = new AluExp(exp.op, exp.dtype, newSrc, exp.arg);
				return visitor(newExp) ?? newExp;
			}
		});
	}
	/** Collect all nodes that satisfy a predicate. */
	collect(predicate) {
		const result = [];
		this.fold((exp) => {
			if (predicate(exp)) result.push(exp);
		});
		return result;
	}
	/** Produce all distinct AluOp in this expression, with their dtypes. */
	distinctOps() {
		const ops = /* @__PURE__ */ new Map();
		this.fold((exp) => {
			const s = ops.get(exp.op) ?? /* @__PURE__ */ new Set();
			if (!s.has(exp.dtype)) {
				s.add(exp.dtype);
				ops.set(exp.op, s);
			}
		});
		return ops;
	}
	/** Rewrite GlobalView operations to GlobalIndex operations. */
	rewriteGlobalViews() {
		return this.rewrite((exp) => {
			if (exp.op === AluOp.GlobalView) {
				const [gid, st] = exp.arg;
				return accessorGlobal(exp.dtype, gid, st, exp.src);
			}
		});
	}
};
/** Symbolic form for each mathematical operation. */
let AluOp = /* @__PURE__ */ function(AluOp$1) {
	AluOp$1["Add"] = "Add";
	AluOp$1["Sub"] = "Sub";
	AluOp$1["Mul"] = "Mul";
	AluOp$1["Idiv"] = "Idiv";
	AluOp$1["Mod"] = "Mod";
	AluOp$1["Min"] = "Min";
	AluOp$1["Max"] = "Max";
	AluOp$1["Sin"] = "Sin";
	AluOp$1["Cos"] = "Cos";
	AluOp$1["Asin"] = "Asin";
	AluOp$1["Atan"] = "Atan";
	AluOp$1["Exp"] = "Exp";
	AluOp$1["Log"] = "Log";
	AluOp$1["Erf"] = "Erf";
	AluOp$1["Erfc"] = "Erfc";
	AluOp$1["Sqrt"] = "Sqrt";
	AluOp$1["Floor"] = "Floor";
	AluOp$1["Ceil"] = "Ceil";
	AluOp$1["Reciprocal"] = "Reciprocal";
	AluOp$1["Cast"] = "Cast";
	AluOp$1["Bitcast"] = "Bitcast";
	AluOp$1["Cmplt"] = "Cmplt";
	AluOp$1["Cmpne"] = "Cmpne";
	AluOp$1["Where"] = "Where";
	AluOp$1["Threefry2x32"] = "Threefry2x32";
	AluOp$1["Const"] = "Const";
	AluOp$1["Special"] = "Special";
	AluOp$1["Variable"] = "Variable";
	AluOp$1["GlobalIndex"] = "GlobalIndex";
	AluOp$1["GlobalView"] = "GlobalView";
	return AluOp$1;
}({});
const AluGroup = {
	Binary: new Set([
		AluOp.Add,
		AluOp.Sub,
		AluOp.Mul,
		AluOp.Idiv,
		AluOp.Mod,
		AluOp.Min,
		AluOp.Max
	]),
	Unary: new Set([
		AluOp.Sin,
		AluOp.Cos,
		AluOp.Asin,
		AluOp.Atan,
		AluOp.Exp,
		AluOp.Log,
		AluOp.Erf,
		AluOp.Erfc,
		AluOp.Sqrt,
		AluOp.Floor,
		AluOp.Ceil,
		AluOp.Reciprocal,
		AluOp.Cast,
		AluOp.Bitcast
	]),
	Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
	Variable: new Set([
		AluOp.Special,
		AluOp.Variable,
		AluOp.GlobalIndex,
		AluOp.GlobalView
	]),
	Reduce: new Set([
		AluOp.Add,
		AluOp.Mul,
		AluOp.Min,
		AluOp.Max
	]),
	RequiredFloat: new Set([
		AluOp.Sin,
		AluOp.Cos,
		AluOp.Asin,
		AluOp.Atan,
		AluOp.Exp,
		AluOp.Log,
		AluOp.Erf,
		AluOp.Erfc,
		AluOp.Sqrt,
		AluOp.Reciprocal,
		AluOp.Floor,
		AluOp.Ceil
	])
};
/** Common variables that can be substituted in expressions. */
const AluVar = {
	gidx: AluExp.variable(DType.Int32, "gidx"),
	ridx: AluExp.variable(DType.Int32, "ridx"),
	acc: (dtype) => AluExp.variable(dtype, "acc"),
	idx: AluExp.variable(DType.Int32, "idx"),
	unroll: AluExp.variable(DType.Int32, "unroll"),
	upcast: AluExp.variable(DType.Int32, "upcast")
};
/**
* Description of a kernel to be compiled.
*
* Each of these can be processed by a backend into some lower-level
* representation. It consists of one or more fused operations, optionally
* indexing into a buffer.
*/
var Kernel = class {
	constructor(nargs, size, exp, reduction) {
		this.nargs = nargs;
		this.size = size;
		this.exp = exp;
		this.reduction = reduction;
		this.exp = exp.simplify();
	}
	hash(state) {
		state.update(this.nargs).update(this.size).update(this.exp).update(this.reduction);
	}
	pprint() {
		let details = PPrint.pp(`exp = ${this.exp}`);
		details = details.concat(PPrint.pp(`size = ${this.size}`));
		if (this.reduction) details = details.concat(PPrint.pp(`reduction = ${this.reduction}`));
		return PPrint.pp("{ ").stack(details).stack(PPrint.pp(" }"));
	}
	toString() {
		return this.pprint().toString();
	}
	/** The dtype of the values output by this kernel. */
	get dtype() {
		if (this.reduction) return this.reduction.epilogue.dtype;
		else return this.exp.dtype;
	}
	/** The number of bytes in the output array when evaluating this kernel. */
	get bytes() {
		return this.size * byteWidth(this.dtype);
	}
};
/**
* Description of a reduction.
*
* The strategy of jax-js backends is to either handle a standard operation that
* is dispatched in a vectorized way over an array, or to reduce over one axis
* of some computation. This is a description of the reduction.
*
* Reduction only supports a few operations, and only over one axis. Users can
* always `flatten()` the array before reducing if needed.
*
* The backend is responsible for implementing the reduction in a way that
* minimizes the number of global memory loads, for efficiency. This involves
* passing through some optimization strategy. But optimizations are not coded
* at this level since they depend on GPU, versus CPU or Wasm.
*/
var Reduction = class Reduction {
	constructor(dtype, op, size, epilogue = AluVar.acc(dtype)) {
		this.dtype = dtype;
		this.op = op;
		this.size = size;
		this.epilogue = epilogue;
		if (!AluGroup.Reduce.has(op)) throw new TypeError(`Unsupported reduction: ${op}`);
		this.epilogue = epilogue.simplify();
	}
	hash(state) {
		state.update(this.dtype).update(this.op).update(this.size).update(this.epilogue);
	}
	toString() {
		return `${this.op}{${this.size}} -> ${this.epilogue}`;
	}
	/** Get the identity for this reduction operation. */
	get identity() {
		if (this.dtype === DType.Bool) return this.op === AluOp.Add || this.op === AluOp.Max ? 0 : 1;
		else if (this.dtype === DType.Int32) {
			if (this.op === AluOp.Add) return 0;
			else if (this.op === AluOp.Mul) return 1;
			else if (this.op === AluOp.Min) return -1 >>> 1;
			else if (this.op === AluOp.Max) return 1 << 31;
		} else if (this.dtype === DType.Uint32) {
			if (this.op === AluOp.Add) return 0;
			else if (this.op === AluOp.Mul) return 1;
			else if (this.op === AluOp.Min) return -1 >>> 0;
			else if (this.op === AluOp.Max) return 0;
		} else if (isFloatDtype(this.dtype)) {
			if (this.op === AluOp.Add) return 0;
			else if (this.op === AluOp.Mul) return 1;
			else if (this.op === AluOp.Min) return Infinity;
			else if (this.op === AluOp.Max) return -Infinity;
		}
		throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
	}
	/** Evaluate this operation on CPU. */
	evaluate(...values) {
		if (this.dtype === DType.Bool) {
			if (this.op === AluOp.Add || this.op === AluOp.Max) return values.reduce((a, b) => a || b, false);
			else if (this.op === AluOp.Mul || this.op === AluOp.Min) return values.reduce((a, b) => a && b, true);
		} else if (this.dtype === DType.Int32) {
			if (this.op === AluOp.Add) return values.reduce((a, b) => a + b | 0, 0);
			else if (this.op === AluOp.Mul) return values.reduce((a, b) => a * b | 0, 1);
			else if (this.op === AluOp.Min) return values.reduce((a, b) => Math.min(a, b), -1 >>> 1);
			else if (this.op === AluOp.Max) return values.reduce((a, b) => Math.max(a, b), 1 << 31);
		} else if (this.dtype === DType.Uint32) {
			if (this.op === AluOp.Add) return values.reduce((a, b) => a + b >>> 0, 0);
			else if (this.op === AluOp.Mul) return values.reduce((a, b) => a * b >>> 0, 1);
			else if (this.op === AluOp.Min) return values.reduce((a, b) => Math.min(a, b), -1 >>> 0);
			else if (this.op === AluOp.Max) return values.reduce((a, b) => Math.max(a, b), 0);
		} else if (isFloatDtype(this.dtype)) {
			if (this.op === AluOp.Add) return values.reduce((a, b) => a + b, 0);
			else if (this.op === AluOp.Mul) return values.reduce((a, b) => a * b, 1);
			else if (this.op === AluOp.Min) return values.reduce((a, b) => Math.min(a, b), Infinity);
			else if (this.op === AluOp.Max) return values.reduce((a, b) => Math.max(a, b), -Infinity);
		}
		throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
	}
	/** Reindex gid values in this reduction's epilogue expression. */
	reindexGids(newGids) {
		const reindexedEpilogue = this.epilogue.reindexGids(newGids);
		if (reindexedEpilogue === this.epilogue) return this;
		return new Reduction(this.dtype, this.op, this.size, reindexedEpilogue);
	}
};
/** Expression for accessing `indices` in input array with the given shape. */
function accessorGlobal(dtype, gid, st, indices) {
	const [index, valid] = st.toAluExp(indices);
	const [, len] = st.views[0].dataRange();
	return AluExp.where(valid, AluExp.globalIndex(dtype, gid, len, index), AluExp.const(dtype, 0));
}
/** Expression for accessing `indices` in an array recipe with variable "idx". */
function accessorAluExp(exp, st, indices) {
	const [index, valid] = st.toAluExp(indices);
	return AluExp.where(valid, exp.substitute({ idx: index }), AluExp.const(exp.dtype, 0));
}
function threefry2x32(k0, k1, c0, c1) {
	const rotl32 = (x, r) => (x << r | x >>> 32 - r) >>> 0;
	const ks0 = k0 >>> 0;
	const ks1 = k1 >>> 0;
	const ks2 = (ks0 ^ ks1 ^ 466688986) >>> 0;
	let x0 = c0 + ks0 >>> 0;
	let x1 = c1 + ks1 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks1 >>> 0;
	x1 = x1 + ks2 + 1 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
	x0 = x0 + ks2 >>> 0;
	x1 = x1 + ks0 + 2 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks0 >>> 0;
	x1 = x1 + ks1 + 3 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 17) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 29) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 16) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 24) ^ x0;
	x0 = x0 + ks1 >>> 0;
	x1 = x1 + ks2 + 4 >>> 0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 13) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 15) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 26) ^ x0;
	x0 = x0 + x1 >>> 0, x1 = rotl32(x1, 6) ^ x0;
	x0 = x0 + ks2 >>> 0;
	x1 = x1 + ks0 + 5 >>> 0;
	return [x0, x1];
}
/**
* Abramowitz & Stegun’s widely used approximation for erf(x).
*
* `erf(x) = 1 - P(t) * exp(-x^2)` for `x >= 0`, where `t = 1/(1 + p*x)` and
* `P(t) = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5`.
*
* Coefficients:
*  - p = 0.3275911
*  - a1 = 0.254829592
*  - a2 = -0.284496736
*  - a3 = 1.421413741
*  - a4 = -1.453152027
*  - a5 = 1.061405429
*
* This function computes just `E = P(t) * exp(-x^2)` for numerical reasons. The
* input is assumed to be non-negative.
*
* Reference: https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
*/
function _erfapprox$1(x) {
	const p = .3275911;
	const a1 = .254829592;
	const a2 = -.284496736;
	const a3 = 1.421413741;
	const a4 = -1.453152027;
	const a5 = 1.061405429;
	const t = 1 / (1 + p * x);
	const P_t = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
	return P_t * Math.exp(-x * x);
}
function erf(x) {
	if (x >= 0) return 1 - _erfapprox$1(x);
	else return _erfapprox$1(-x) - 1;
}
function erfc(x) {
	if (x >= 0) return _erfapprox$1(x);
	else return 2 - _erfapprox$1(-x);
}

//#endregion
//#region src/routine.ts
/**
* Advanced operations that don't fit into the `AluExp` compiler representation.
*
* Some routines like iterative matrix algorithms, FFTs, or sorting may not be
* easy to express efficiently as a `Kernel` object. These also tend to be
* somewhat expensive, so the benefit of kernel fusion and inlining is less
* relevant.
*
* For these operations, we dispatch them as a custom operation on the backend,
* which each backend implements in a specific way. These are listed in the
* `Routines` enum below.
*
* Routines cannot be fused into other kernels and always operate on contiguous
* arrays (default `ShapeTracker`).
*/
var Routine = class {
	constructor(name, type, params) {
		this.name = name;
		this.type = type;
		this.params = params;
	}
};
/** One of the valid `Routine` that can be dispatched to backend. */
let Routines = /* @__PURE__ */ function(Routines$1) {
	/** Stable sorting algorithm along the last axis. */
	Routines$1["Sort"] = "Sort";
	/** Returns `int32` indices of the stably sorted array. */
	Routines$1["Argsort"] = "Argsort";
	/**
	* Solve a triangular system of equations.
	*
	* The first batch of inputs `A` should be of shape `[..., N, N]` and upper
	* triangular, while the second batch `B` should be of shape `[..., M, N]`.
	*
	* Solves for `X` in the equation `A @ X.T = B.T`, where `A` is the
	* triangular matrix. This is equivalent to `X = B @ A^-T`.
	*/
	Routines$1["TriangularSolve"] = "TriangularSolve";
	/**
	* Cholesky decomposition of 2D positive semi-definite matrices.
	*
	* The input batch should be of shape `[..., N, N]`, and the output batch is
	* of the same shape, containing the lower-triangular matrix `L` such that
	* `A = L @ L.T`. Behavior is unspecified if A is not positive semi-definite.
	*/
	Routines$1["Cholesky"] = "Cholesky";
	/**
	* LU decomposition of 2D rectangular matrices.
	*
	* The input is a batch of shape `[..., M, N]`, and the output is a tuple of
	* three arrays: `LU, Pivots, Permutation`.
	*
	* - `LU` is of shape `[..., M, N]`, containing the combined lower and upper
	*   triangular matrices. (lower triangular = implicit unit diagonal)
	* - `Pivots` is of shape `[..., min(M, N)]`, containing the row swaps.
	* - `Permutation` is of shape `[..., M]`, containing the permutation vector
	*   such that `P = eye(M).slice(Permutation)` -> `P @ A = L @ U`.
	*/
	Routines$1["LU"] = "LU";
	return Routines$1;
}({});
function runCpuRoutine(routine, inputs, outputs) {
	const { name, type } = routine;
	const inputAr = inputs.map((buf, i) => dtypedArray(type.inputDtypes[i], buf));
	const outputAr = outputs.map((buf, i) => dtypedArray(type.outputDtypes[i], buf));
	switch (name) {
		case Routines.Sort: return runSort(type, inputAr, outputAr);
		case Routines.Argsort: return runArgsort(type, inputAr, outputAr);
		case Routines.TriangularSolve: return runTriangularSolve(type, inputAr, outputAr, routine.params);
		case Routines.Cholesky: return runCholesky(type, inputAr, outputAr);
		case Routines.LU: return runLU(type, inputAr, outputAr);
		default:
	}
}
function runSort(type, [x], [y]) {
	const xs = type.inputShapes[0];
	if (xs.length === 0) throw new Error("sort: cannot sort a scalar");
	const n = xs[xs.length - 1];
	y.set(x);
	for (let i = 0; i < y.length; i += n) y.subarray(i, i + n).sort();
}
function runArgsort(type, [x], [y, yi]) {
	const xs = type.inputShapes[0];
	if (xs.length === 0) throw new Error("argsort: cannot sort a scalar");
	const n = xs[xs.length - 1];
	for (let offset = 0; offset < y.length; offset += n) {
		const ar = x.subarray(offset, offset + n);
		const out = y.subarray(offset, offset + n);
		const outi = yi.subarray(offset, offset + n);
		for (let i = 0; i < n; i++) outi[i] = i;
		outi.sort((a, b) => ar[a] - ar[b]);
		for (let i = 0; i < n; i++) out[i] = ar[outi[i]];
	}
}
function runTriangularSolve(type, [a, b], [x], { unitDiagonal }) {
	const as = type.inputShapes[0];
	const bs = type.inputShapes[1];
	if (as.length < 2) throw new Error(`triangular_solve: a must be at least 2D, got ${as}`);
	if (bs.length < 2) throw new Error(`triangular_solve: b must be at least 2D, got ${bs}`);
	const n = as[as.length - 2];
	if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
	const batch = bs[bs.length - 2];
	for (let counter = 0; counter < a.length / (n * n); counter++) {
		const a1 = a.subarray(counter * n * n, (counter + 1) * n * n);
		for (let t = 0; t < batch; t++) {
			const b1 = b.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
			const x1 = x.subarray((counter * batch + t) * n, (counter * batch + t + 1) * n);
			for (let i = n - 1; i >= 0; i--) {
				let sum = b1[i];
				for (let j = i + 1; j < n; j++) sum -= a1[i * n + j] * x1[j];
				x1[i] = unitDiagonal ? sum : sum / a1[i * n + i];
			}
		}
	}
}
function runCholesky(type, [x], [y]) {
	const xs = type.inputShapes[0];
	if (xs.length < 2) throw new Error("cholesky: input must be at least 2D");
	const n = xs[xs.length - 2];
	const m = xs[xs.length - 1];
	if (n !== m) throw new Error(`cholesky: input must be square, got [${n}, ${m}]`);
	for (let offset = 0; offset < y.length; offset += n * n) {
		const ar = x.subarray(offset, offset + n * n);
		const out = y.subarray(offset, offset + n * n);
		for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
			let sum = ar[i * n + j];
			for (let k = 0; k < j; k++) sum -= out[i * n + k] * out[j * n + k];
			out[i * n + j] = i === j ? Math.sqrt(sum) : sum / out[j * n + j];
		}
	}
}
function runLU(type, [a], [lu, pivots, perm]) {
	const shape = type.inputShapes[0];
	if (shape.length < 2) throw new Error("lu: input must be at least 2D");
	const m = shape[shape.length - 2];
	const n = shape[shape.length - 1];
	const r = Math.min(m, n);
	for (let offset = 0; offset < a.length; offset += m * n) {
		const ar = a.subarray(offset, offset + m * n);
		const out = lu.subarray(offset, offset + m * n);
		const batchIdx = offset / (m * n);
		const piv = pivots.subarray(batchIdx * r, (batchIdx + 1) * r);
		const p = perm.subarray(batchIdx * m, (batchIdx + 1) * m);
		out.set(ar);
		for (let i = 0; i < m; i++) p[i] = i;
		for (let j = 0; j < r; j++) {
			let maxVal = Math.abs(out[j * n + j]);
			let maxRow = j;
			for (let i = j + 1; i < m; i++) {
				const val = Math.abs(out[i * n + j]);
				if (val > maxVal) {
					maxVal = val;
					maxRow = i;
				}
			}
			piv[j] = maxRow;
			if (maxRow !== j) {
				for (let col = 0; col < n; col++) {
					const tmp = out[j * n + col];
					out[j * n + col] = out[maxRow * n + col];
					out[maxRow * n + col] = tmp;
				}
				const tmpP = p[j];
				p[j] = p[maxRow];
				p[maxRow] = tmpP;
			}
			const diag = out[j * n + j];
			if (diag !== 0) for (let i = j + 1; i < m; i++) {
				const factor = out[i * n + j] / diag;
				out[i * n + j] = factor;
				for (let col = j + 1; col < n; col++) out[i * n + col] -= factor * out[j * n + col];
			}
		}
	}
}

//#endregion
//#region src/shape.ts
const jstr = JSON.stringify;
/** Remove "1" dimensions from the strides list. */
function canonicalizeStrides(shape, strides) {
	const newStrides = [];
	for (let i = 0; i < shape.length; i++) if (shape[i] === 1) newStrides.push(0);
	else newStrides.push(strides[i]);
	return newStrides;
}
/** Get the strides for a shape in default row-major order. */
function defaultStrides(shape) {
	if (shape.length === 0) return [];
	const strides = rep(shape.length, 1);
	for (let i = shape.length - 1; i > 0; i--) strides[i - 1] = shape[i] * strides[i];
	return canonicalizeStrides(shape, strides);
}
/** Merge contiguous subparts or zero-strided dimensions in a view. */
function mergeDims(shape, strides, mask) {
	if (shape.length === 0) return [];
	if (shape.length !== strides.length || mask && shape.length !== mask.length) throw new Error("internal: invalid args to mergeDims");
	const ret = [[
		shape[0],
		strides[0],
		strides[0] !== 0 ? shape[0] : 0
	]];
	let merging = mask ? mask[0][1] - mask[0][0] === 1 : shape[0] === 1;
	for (let i = 1; i < shape.length; i++) {
		const [s, st] = [shape[i], strides[i]];
		if (s === 1) continue;
		const [lastS, lastSt, lastPreExpandS] = ret[ret.length - 1];
		if (merging || lastSt === s * st) ret[ret.length - 1] = [
			lastS * s,
			st,
			merging ? s : lastPreExpandS * s
		];
		else ret.push([
			s,
			st,
			s
		]);
		merging = mask ? mask[i][1] - mask[i][0] === 1 : false;
	}
	return ret;
}
/** Return the new mask if a reshape if possible, otherwise `null`. */
function reshapeMask(maskInput, oldShape, newShape) {
	const newMask = [];
	let rMasksI = maskInput.length;
	let rShapeI = oldShape.length;
	let rNewShapeI = newShape.length;
	const rMasks = () => rMasksI ? maskInput[--rMasksI] : [0, 1];
	const rShape = () => rShapeI ? oldShape[--rShapeI] : 1;
	const rNewShape = () => rNewShapeI ? newShape[--rNewShapeI] : 1;
	let currStride = 1;
	let [oldDim, newDim, mask] = [
		rShape(),
		rNewShape(),
		rMasks()
	];
	while (newMask.length < newShape.length) {
		const [l, r] = mask;
		const nextStride = newDim * currStride;
		if (oldDim === nextStride) {
			newMask.push([intdiv(l, currStride), intdiv(r - 1, currStride) + 1]);
			currStride = 1;
			[oldDim, newDim, mask] = [
				rShape(),
				rNewShape(),
				rMasks()
			];
		} else if (oldDim > nextStride) {
			if (oldDim % nextStride !== 0) return null;
			if ((l % nextStride !== 0 || r % nextStride !== 0) && intdiv(l, nextStride) !== intdiv(r - 1, nextStride)) return null;
			newMask.push([intdiv(l % nextStride, currStride), intdiv((r - 1) % nextStride, currStride) + 1]);
			[currStride, newDim] = [nextStride, rNewShape()];
		} else {
			const nextMask = rMasks();
			if (!deepEqual(mask, [0, oldDim]) && l !== r && nextMask[1] - nextMask[0] !== 1) return null;
			mask = [nextMask[0] * oldDim + l, (nextMask[1] - 1) * oldDim + r];
			oldDim *= rShape();
		}
	}
	return newMask.reverse();
}
/**
* A multidimensional view into memory. An array can be thought of as the
* combination of a linear buffer of memory, along with a `View`.
*
* Formula for getting a data point is basically:
*   1. Check if ∀i. 0 <= dim[i] < shape[i], otherwise out of bounds.
*   2. If mask exists, and ∃i. dim[i] ∉ mask[i], return 0.
*   2. Otherwise, look at this memory address: offset + ∑(strides[i] * dim[i]).
*/
var View = class View {
	#size;
	#contiguous;
	constructor(shape, strides, offset, mask) {
		this.shape = shape;
		this.strides = strides;
		this.offset = offset;
		this.mask = mask;
	}
	static create(shape, strides, offset = 0, mask = null) {
		if (shape.some((s) => s < 0)) throw new Error("View shape must be non-negative");
		strides = strides ? canonicalizeStrides(shape, strides) : defaultStrides(shape);
		if (shape.includes(0)) return new View(shape, rep(shape.length, 0), 0, null);
		if (mask !== null && mask.every(([b, e], i) => b === 0 && e === shape[i])) mask = null;
		if (mask !== null) {
			const elimDims = [];
			let hasNoData = false;
			for (let i = 0; i < shape.length; i++) {
				const [b, e] = mask[i];
				if (b + 1 >= e) elimDims.push(i);
				if (b >= e) hasNoData = true;
			}
			if (elimDims.length) {
				if (hasNoData) {
					strides = rep(shape.length, 0);
					offset = 0;
					mask = rep(shape.length, () => [0, 0]);
				}
				for (const i of elimDims) {
					offset += strides[i] * mask[i][0];
					strides[i] = 0;
				}
			}
		}
		return new View(shape, strides, offset, mask);
	}
	get ndim() {
		return this.shape.length;
	}
	get size() {
		if (this.#size === void 0) this.#size = prod(this.shape);
		return this.#size;
	}
	/** Whether this is a default, contiguous, unaltered view of the data (identity). */
	get contiguous() {
		if (this.#contiguous === void 0) this.#contiguous = this.size === 0 || this.offset === 0 && this.mask === null && deepEqual(this.strides, defaultStrides(this.shape));
		return this.#contiguous;
	}
	/** Return the range of data being indexed in this view, or [0, 0] if none. */
	dataRange() {
		if (this.size === 0 || this.mask && this.mask[0][0] === this.mask[0][1]) return [0, 0];
		let min = this.offset;
		let max = this.offset;
		for (let i = 0; i < this.ndim; i++) {
			let [lo, hi] = this.mask ? this.mask[i] : [0, this.shape[i]];
			--hi;
			const s = this.strides[i];
			if (s > 0) {
				min += s * lo;
				max += s * hi;
			} else if (s < 0) {
				min += s * hi;
				max += s * lo;
			}
		}
		return [min, max + 1];
	}
	/** Produce an AluExp for evaluating this view at an index. */
	toAluExp(idxs) {
		let iexpr = AluExp.i32(this.offset);
		let vexpr = AluExp.bool(true);
		for (let i = this.ndim - 1; i >= 0; i--) {
			const idx = idxs[i];
			if (this.shape[i] !== 1 && this.strides[i] !== 0) iexpr = AluExp.add(AluExp.mul(idx, AluExp.i32(this.strides[i])), iexpr);
			if (this.mask) {
				if (this.mask[i][0] !== 0) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][0])).not(), vexpr);
				if (this.mask[i][1] !== this.shape[i]) vexpr = AluExp.mul(AluExp.cmplt(idx, AluExp.i32(this.mask[i][1])), vexpr);
			}
		}
		return [iexpr, vexpr];
	}
	/**
	* Try to compose this view with another one. `this` view is applied first,
	* followed by the argument. If this is not possible for the specific views,
	* return `null` instead.
	*
	* If composable, return a combined view with the same shape as `v1`.
	*
	* This is very tricky. The shapes of v1 and v2 may be different, and in that
	* case, we do some math to figure out whether they're compatible.
	*/
	compose(v1) {
		const v2 = this;
		if (v2.contiguous) return v1;
		if (v1.contiguous) {
			if (deepEqual(v1.shape, v2.shape)) return v2;
			if (v1.size === v2.size) {
				const ret = v2.reshape(v1.shape);
				if (ret !== null) return ret;
			}
		}
		if (v1.mask !== null) {
			const newV1 = v1.shrink(v1.mask);
			const merged = v2.compose(newV1);
			return merged ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]])) : null;
		}
		const origin = unravel(v2.shape, v1.offset);
		const terms = rep(v2.ndim, () => []);
		const strides = rep(v1.ndim, 0);
		for (let d1 = 0; d1 < v1.strides.length; d1++) {
			const st = v1.strides[d1];
			if (st === 0) continue;
			const unravelOffset = unravel(v2.shape, v1.offset + st);
			for (let d2 = 0; d2 < v2.ndim; d2++) {
				const o = origin[d2];
				const diff = unravelOffset[d2] - o;
				if (diff === 0) continue;
				terms[d2].push([d1, diff]);
				strides[d1] += diff * v2.strides[d2];
			}
		}
		let [mergedSize, mergedTermMin, mergedTermMax] = [
			1,
			0,
			0
		];
		const extents = [];
		for (let i = v2.ndim - 1; i >= 0; i--) {
			const term = terms[i];
			const s = v2.shape[i];
			let [tmin, tmax] = [origin[i], origin[i]];
			for (const [d1, s1] of term) if (s1 > 0) tmax += (v1.shape[d1] - 1) * s1;
			else if (s1 < 0) tmin += (v1.shape[d1] - 1) * s1;
			mergedTermMin += tmin * mergedSize;
			mergedTermMax += tmax * mergedSize;
			mergedSize *= s;
			if (mergedTermMin >= 0 && mergedTermMax < mergedSize) {
				extents.push([
					mergedSize,
					mergedTermMin,
					mergedTermMax
				]);
				[mergedSize, mergedTermMin, mergedTermMax] = [
					1,
					0,
					0
				];
			}
		}
		if (mergedTermMin !== 0 || mergedTermMax !== 0) return null;
		extents.reverse();
		const v2Shape = extents.map(([s]) => s);
		if (!deepEqual(v2Shape, v2.shape)) {
			const reshapedV2 = v2.reshape(v2Shape);
			if (reshapedV2 === null) return null;
			if (!deepEqual(reshapedV2.shape, v2.shape)) return reshapedV2.compose(v1);
		}
		if (v2.mask !== null) {
			const newB = rep(v1.ndim, 0);
			const newE = v1.shape.slice();
			let bad = false;
			for (let d2 = 0; d2 < v2.ndim; d2++) {
				const [b, e] = v2.mask[d2];
				const o = origin[d2];
				const term = terms[d2];
				const [_, tmin, tmax] = extents[d2];
				if (b <= tmin && tmax < e) continue;
				if (term.length !== 1) if (term.length === 0 && newE.length) newE[0] = 0;
				else bad = true;
				else {
					const [d1, s1] = term[0];
					newB[d1] = Math.max(newB[d1], Math.ceil((s1 > 0 ? b - o : e - o - 1) / s1));
					newE[d1] = Math.min(newE[d1], Math.floor((s1 < 0 ? b - o : e - o - 1) / s1) + 1);
				}
			}
			for (let d1 = 0; d1 < v1.ndim; d1++) if (newB[d1] !== 0 || newE[d1] !== v1.shape[d1]) return v2.compose(View.create(v1.shape, v1.strides, v1.offset, zip(newB, newE)));
			if (bad) return null;
		}
		let finalOffset = v2.offset;
		for (let d2 = 0; d2 < v2.ndim; d2++) finalOffset += origin[d2] * v2.strides[d2];
		return View.create(v1.shape, strides, finalOffset, null);
	}
	/** Attempt to simplify this view into a smaller reshaped form. */
	minify() {
		const minShape = mergeDims(this.shape, this.strides, this.mask).map((x) => x[0]);
		const nv = this.reshape(minShape);
		return nv ? nv : this;
	}
	/** Pad the view with zeros on each dimension. */
	pad(arg) {
		if (arg.length !== this.ndim || !arg.every(([b, e]) => b >= 0 && e >= 0)) throw new Error(`invalid pad ${jstr(arg)} for ${jstr(this.shape)}`);
		if (arg.every(([b, e]) => b === 0 && e === 0)) return this;
		const zvarg = arg.map(([b, e], i) => [-b, this.shape[i] + e]);
		const mask = arg.map(([b, _e], i) => [b, this.shape[i] + b]);
		return this.#unsafeResize(zvarg, mask);
	}
	/** Shrink the view by taking a subarray. */
	shrink(arg) {
		if (arg.length !== this.ndim || !arg.every(([b, e], i) => 0 <= b && b <= e && e <= this.shape[i])) throw new Error(`invalid shrink ${jstr(arg)} for ${jstr(this.shape)}`);
		return this.#unsafeResize(arg);
	}
	#unsafeResize(arg, mask) {
		const offset = this.strides.map((s, i) => s * arg[i][0]).reduce((a, b) => a + b, 0);
		if (this.mask) {
			const nmask = this.mask.map(([mx, my], i) => [Math.max(0, Math.min(mx - arg[i][0], arg[i][1] - arg[i][0])), Math.max(0, Math.min(my - arg[i][0], arg[i][1] - arg[i][0]))]);
			mask = mask ? mask.map(([mx, my], i) => [Math.max(mx, nmask[i][0]), Math.min(my, nmask[i][1])]) : nmask;
		}
		return View.create(arg.map(([b, e]) => e - b), this.strides, this.offset + offset, mask);
	}
	/** Expand one or more axes with length "1" by repeating the data. */
	expand(newShape) {
		if (newShape.length !== this.ndim) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
		for (let i = 0; i < this.ndim; i++) if (newShape[i] !== this.shape[i] && this.shape[i] !== 1) throw new Error(`Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`);
		if (this.size === 0) return View.create(newShape);
		const mask = this.mask ? this.mask.map((m, i) => this.shape[i] === newShape[i] ? m : m[0] === 0 && m[1] === 1 ? [0, newShape[i]] : [0, 0]) : null;
		return View.create(newShape, this.strides, this.offset, mask);
	}
	/** Permute the axes of an array. */
	permute(axis) {
		if (!isPermutation(axis, this.ndim)) throw new Error(`Invalid permutation ${jstr(axis)} of len ${this.ndim}`);
		const newShape = axis.map((a) => this.shape[a]);
		const newStrides = axis.map((a) => this.strides[a]);
		const newMask = this.mask ? axis.map((a) => this.mask[a]) : null;
		return View.create(newShape, newStrides, this.offset, newMask);
	}
	/** Flip (reverse) one or more axes of the view. */
	flip(arg) {
		if (arg.length !== this.ndim) throw new Error(`Invalid flip ${jstr(arg)} for ${jstr(this.shape)}`);
		const strides = this.strides.slice();
		let offset = this.offset;
		const mask = this.mask ? this.mask.slice() : null;
		for (let i = 0; i < this.ndim; i++) {
			const s = this.shape[i];
			if (arg[i]) {
				strides[i] = -strides[i];
				offset += (s - 1) * this.strides[i];
				if (mask) mask[i] = [s - mask[i][1], s - mask[i][0]];
			}
		}
		return View.create(this.shape, strides, offset, mask);
	}
	/** Reshape the view into a new shape. */
	reshape(newShape) {
		if (deepEqual(this.shape, newShape)) return this;
		if (newShape.some((s) => s < 0)) throw new Error(`Reshape cannot have negative numbers ${jstr(newShape)}`);
		if (this.size !== prod(newShape)) throw new Error(`Reshape size ${jstr(this.shape)} -> ${jstr(newShape)}`);
		if (this.size === 0) return View.create(newShape);
		if (newShape.length === 0 && this.mask?.some(([b, e]) => b === e)) return null;
		if (this.contiguous) return View.create(newShape);
		const rStrides = [];
		const merge = mergeDims(this.shape, this.strides, this.mask);
		let rShapeIdx = newShape.length;
		for (let i = merge.length - 1; i >= 0; i--) {
			let [mergedSize, newStride, realSize] = merge[i];
			let acc = 1;
			while (acc < mergedSize && rShapeIdx > 0) {
				const newDim = newShape[--rShapeIdx];
				rStrides.push(newStride * acc);
				acc *= newDim;
				if (acc >= realSize) newStride = 0;
			}
			if (acc !== mergedSize) return null;
		}
		const newStrides = rep(newShape.length - rStrides.length, 0).concat(rStrides.reverse());
		if (!this.mask) return View.create(newShape, newStrides, this.offset);
		const newMask = reshapeMask(this.mask, this.shape, newShape);
		if (!newMask) return null;
		let newOffset = this.offset;
		for (let i = 0; i < this.ndim; i++) newOffset += this.strides[i] * this.mask[i][0];
		for (let i = 0; i < newShape.length; i++) newOffset -= newStrides[i] * newMask[i][0];
		return View.create(newShape, newStrides, newOffset, newMask);
	}
};
/**
* Find position of `offset` in each dimension within an existing shape. Like
* `numpy.unravel_index` in behavior.
*/
function unravel(shape, offset) {
	let acc = 1;
	const idxs = [];
	for (let i = shape.length - 1; i >= 0; i--) {
		const d = shape[i];
		idxs.push(Math.floor(offset / acc) % d);
		acc *= d;
	}
	return idxs.reverse();
}
/** Generate a list of AluExp for computing unravel(). */
function unravelAlu(shape, offset) {
	let acc = 1;
	const idxs = [];
	for (let i = shape.length - 1; i >= 0; i--) {
		const d = shape[i];
		idxs.push(AluExp.mod(AluExp.idiv(offset, AluExp.i32(acc)), AluExp.i32(d)));
		acc *= d;
	}
	return idxs.reverse();
}
/**
* Array shape after applying movement operations, as a series of views.
*
* Each view is applied, then treated as if it were a contiguous array of its
* shape, then used as the virtual buffer for the next view.
*/
var ShapeTracker = class ShapeTracker {
	constructor(views) {
		this.views = views;
	}
	/** Compose this shape tracker with another, applying it after this one. */
	compose(other) {
		if (this.contiguous) return other;
		let ret = this;
		for (const v of other.views) ret = new ShapeTracker(ret.views.concat(v)).simplify();
		return ret;
	}
	static fromShape(shape) {
		return new ShapeTracker([View.create(shape)]);
	}
	get contiguous() {
		return this.views.length === 1 && this.views[0].contiguous;
	}
	get consecutive() {
		return this.views.length === 1 && this.views[0].mask === null && deepEqual(this.views[0].strides, defaultStrides(this.views[0].shape));
	}
	get lastStrides() {
		return this.views[this.views.length - 1].strides;
	}
	get shape() {
		return this.views[this.views.length - 1].shape;
	}
	get size() {
		return this.views[this.views.length - 1].size;
	}
	toAluExp(idxs) {
		let [iexpr, vexpr] = this.views[this.views.length - 1].toAluExp(idxs);
		for (let i = this.views.length - 2; i >= 0; i--) {
			const view = this.views[i].minify();
			const exprs = view.toAluExp(unravelAlu(view.shape, iexpr));
			iexpr = exprs[0];
			vexpr = AluExp.mul(vexpr, exprs[1]);
		}
		return [iexpr.simplify(), vexpr.simplify()];
	}
	simplify() {
		const views = this.views.slice();
		while (views.length >= 2) {
			const newView = views[views.length - 2].compose(views[views.length - 1]);
			if (newView === null) break;
			views.splice(views.length - 2, 2, newView);
		}
		return new ShapeTracker(views);
	}
	pad(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.pad(arg)));
	}
	shrink(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.shrink(arg)));
	}
	expand(newShape) {
		return new ShapeTracker(applyLast(this.views, (x) => x.expand(newShape)));
	}
	permute(axis) {
		return new ShapeTracker(applyLast(this.views, (x) => x.permute(axis)));
	}
	flip(arg) {
		return new ShapeTracker(applyLast(this.views, (x) => x.flip(arg)));
	}
	reshape(newShape) {
		const newView = this.views[this.views.length - 1].reshape(newShape);
		return new ShapeTracker(newView === null ? this.views.concat(View.create(newShape)) : this.views.toSpliced(this.views.length - 1, 1, newView));
	}
	/** Broadcast along the given new axes, then expand the shape. */
	broadcast(newShape, axis) {
		let st = this;
		if (axis.length > 0) {
			const unsqueezed = [...st.shape];
			for (const i of sorted(axis)) unsqueezed.splice(i, 0, 1);
			st = st.reshape(unsqueezed);
		}
		return st.expand(newShape);
	}
	/**
	* Repeat data in each axis by a positive number of repetitions.
	*
	* - If `tile` is true (default): [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
	* - If `tile` is false: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].
	*/
	repeat(reps, tile = true) {
		if (reps.length > this.shape.length) throw new Error(`Too many repeats ${jstr(reps)} for shape ${jstr(this.shape)}`);
		if (reps.some((c) => c <= 0)) throw new Error(`Invalid repeats ${jstr(reps)}`);
		if (reps.length === 0) return this;
		const noop = this.shape.slice(0, -reps.length);
		const shape = this.shape.slice(-reps.length);
		return this.broadcast([...noop, ...shape.flatMap((s, i) => tile ? [reps[i], s] : [s, reps[i]])], shape.map((_, i) => noop.length + 2 * i + (tile ? 0 : 1))).reshape([...noop, ...shape.map((s, i) => s * reps[i])]);
	}
	/** Move axis i to axis j. */
	moveaxis(i, j) {
		const perm = range(this.shape.length);
		perm.splice(i, 1);
		perm.splice(j, 0, i);
		return this.permute(perm);
	}
	/** Like pad(), but allows for negative values. */
	padOrShrink(arg) {
		const padArg = [];
		const shrinkArg = [];
		for (let i = 0; i < arg.length; i++) {
			const [b, e] = arg[i];
			if (b < -this.shape[i] || e < -this.shape[i] || b + e < -this.shape[i]) throw new Error(`Invalid padOrShrink ${jstr(arg)} for ${jstr(this.shape)}`);
			padArg.push([Math.max(0, b), Math.max(0, e)]);
			shrinkArg.push([Math.max(0, -b), this.shape[i] - Math.max(0, -e)]);
		}
		return this.shrink(shrinkArg).pad(padArg);
	}
};
function applyLast(ar, f) {
	return ar.toSpliced(ar.length - 1, 1, f(ar[ar.length - 1]));
}

//#endregion
//#region src/tuner.ts
/** Stores dimensions of the kernel's applied shape. Globals start at 0. */
var TuneDims = class {
	st;
	outputSt;
	groups;
	reduce;
	unroll;
	upcast;
	get end() {
		return this.st.shape.length;
	}
	constructor(shape) {
		this.st = ShapeTracker.fromShape(shape);
		this.outputSt = ShapeTracker.fromShape(shape.slice(0, -1));
		this.groups = this.st.shape.length - 1;
		this.reduce = this.st.shape.length - 1;
		this.unroll = this.st.shape.length;
		this.upcast = this.st.shape.length;
	}
	applyLocal(axis, amount) {
		if (axis >= this.groups) throw new Error("Cannot localize reduction axis");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Localize by ${amount} on axis length ${length}`);
		if (length !== amount) {
			this.groups++, this.reduce++, this.unroll++, this.upcast++;
			this.st = this.st.reshape([
				...this.st.shape.slice(0, axis),
				length / amount,
				amount,
				...this.st.shape.slice(axis + 1)
			]);
			this.outputSt = this.outputSt.reshape([
				...this.outputSt.shape.slice(0, axis),
				length / amount,
				amount,
				...this.outputSt.shape.slice(axis + 1)
			]);
			axis++;
		}
		this.st = this.st.permute([
			...range(axis),
			...range(axis + 1, this.groups),
			axis,
			...range(this.groups, this.st.shape.length)
		]);
		this.outputSt = this.outputSt.permute([
			...range(axis),
			...range(axis + 1, this.groups),
			axis,
			...range(this.groups, this.outputSt.shape.length)
		]);
	}
	applyUpcast(axis, amount) {
		if (axis >= this.groups) throw new Error("Cannot upcast along reduction axis");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Upcast by ${amount} on axis length ${length}`);
		this.st = this.st.reshape([
			...this.st.shape.slice(0, axis),
			length / amount,
			amount,
			...this.st.shape.slice(axis + 1)
		]).permute([
			...range(axis + 1),
			...range(axis + 2, this.st.shape.length + 1),
			axis + 1
		]);
		this.outputSt = this.outputSt.reshape([
			...this.outputSt.shape.slice(0, axis),
			length / amount,
			amount,
			...this.outputSt.shape.slice(axis + 1)
		]).permute([
			...range(axis + 1),
			...range(axis + 2, this.outputSt.shape.length + 1),
			axis + 1
		]);
	}
	applyUnroll(axis, amount) {
		if (axis < this.groups) throw new Error("Cannot unroll non-reduce axis");
		if (axis >= this.unroll) throw new Error("Axis already unrolled");
		const length = this.st.shape[axis];
		if (length % amount !== 0) throw new Error(`Unroll by ${amount} on axis length ${length}`);
		if (length === amount) {
			this.st = this.st.permute([
				...range(axis),
				...range(axis + 1, this.upcast),
				axis,
				...range(this.upcast, this.st.shape.length)
			]);
			if (axis < this.reduce) this.reduce--;
			this.unroll--;
		} else {
			this.st = this.st.reshape([
				...this.st.shape.slice(0, axis),
				length / amount,
				amount,
				...this.st.shape.slice(axis + 1)
			]).permute([
				...range(axis + 1),
				...range(axis + 2, this.upcast + 1),
				axis + 1,
				...range(this.upcast + 1, this.st.shape.length + 1)
			]);
			this.upcast++;
		}
	}
};
/** Tuning step that does not apply any optimization. */
function tuneNullopt(kernel) {
	const vars = {};
	vars.gidx = AluExp.special(DType.Int32, "gidx", kernel.size);
	if (kernel.reduction) vars.ridx = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
	return {
		exp: kernel.exp.substitute(vars).rewriteGlobalViews().simplify(),
		epilogue: kernel.reduction?.epilogue.substitute({ gidx: vars.gidx }).rewriteGlobalViews().simplify(),
		outputIdxExp: vars.gidx,
		threadCount: kernel.size,
		size: { reduce: kernel.reduction ? kernel.reduction.size : 0 }
	};
}
/** Tuning for WebGPU kernels. */
function tuneWebgpu(kernel) {
	const { exp, reduction } = kernel;
	if (!reduction) return tuneNullopt(kernel);
	const globalIndexes = exp.collect((exp$1) => exp$1.op === AluOp.GlobalIndex);
	if (globalIndexes.length > 0) {
		if (DEBUG >= 4) console.info("Tuning: Found GlobalIndex ops, skipping opt.");
		return tuneNullopt(kernel);
	}
	const globalViews = exp.collect((exp$1) => exp$1.op === AluOp.GlobalView);
	if (globalViews.length === 0) {
		if (DEBUG >= 4) console.info("Tuning: No GlobalView ops found in kernel.");
		return tuneNullopt(kernel);
	}
	const shape = globalViews[0].arg[1].shape;
	const expectedSrc = [...unravelAlu(shape.slice(0, -1), AluVar.gidx), AluVar.ridx].map((e) => e.simplify());
	for (const gv of globalViews) if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
		if (DEBUG >= 4) console.info("Tuning: GlobalView src[] not consistent with reduction.");
		return tuneNullopt(kernel);
	}
	if (shape[shape.length - 1] !== reduction.size) throw new Error("Invariant violation: shape doesn't match reduction size.");
	const sts = globalViews.map((gv) => gv.arg[1]);
	for (const st of sts) if (!deepEqual(st.shape, shape)) throw new Error("Invariant violation: GlobalView shape mismatch");
	const dim = new TuneDims(shape);
	const upcastedAxis = /* @__PURE__ */ new Set();
	while (prod(dim.st.shape.slice(0, dim.groups)) >= 1024) {
		const choices = [];
		const composedSts = sts.map((st) => st.compose(dim.st));
		for (let axis = 0; axis < dim.groups; axis++) for (const amount of [
			3,
			4,
			5
		]) if (!upcastedAxis.has(axis) && dim.st.shape[axis] % amount === 0 && composedSts.some((st) => st.lastStrides[axis] === 0 && st.lastStrides.slice(dim.unroll).every((stride) => stride > 0))) {
			let nonzeroStrides = 0;
			let totalStrides = 0;
			for (const st of composedSts) {
				nonzeroStrides += st.lastStrides[axis] > 0 ? 1 : 0;
				totalStrides += st.lastStrides[axis];
			}
			choices.push([
				nonzeroStrides,
				totalStrides,
				axis,
				amount
			]);
		}
		if (choices.length > 0) {
			choices.sort(lexCompare);
			dim.applyUpcast(choices[0][2], choices[0][3]);
			upcastedAxis.add(choices[0][2]);
		} else break;
	}
	if (!/Mobi|Android/i.test(navigator.userAgent) && dim.reduce < dim.unroll && (prod(dim.st.shape.slice(dim.unroll)) <= 4 || dim.unroll === dim.upcast && prod(dim.st.shape.slice(dim.upcast)) < 64)) {
		const s = dim.st.shape[dim.unroll - 1];
		if (0 < s && s <= 32) dim.applyUnroll(dim.reduce, s);
		else for (const splits of [8, 4]) if (s % splits === 0) {
			dim.applyUnroll(dim.unroll - 1, splits);
			break;
		}
	}
	for (const ax of sorted(upcastedAxis)) {
		const s = dim.st.shape[ax];
		for (const amount of [8, 4]) if (s % amount === 0) {
			dim.applyLocal(ax, amount);
			break;
		}
	}
	const indices = [];
	const addIndices = (s, exp$1) => {
		if (s.length === 0) return;
		else if (s.length === 1) indices.push(exp$1);
		else indices.push(...unravelAlu(s, exp$1));
	};
	if (0 < dim.groups) {
		const s = dim.st.shape.slice(0, dim.groups);
		addIndices(s, AluExp.special(DType.Int32, "gidx", prod(s)));
	}
	if (dim.groups < dim.reduce) {
		const s = dim.st.shape.slice(dim.groups, dim.reduce);
		addIndices(s, AluExp.special(DType.Int32, "group", prod(s)));
	}
	if (dim.reduce <= dim.unroll) {
		const s = dim.st.shape.slice(dim.reduce, dim.unroll);
		addIndices(s, AluExp.special(DType.Int32, "ridx", prod(s)));
	}
	if (dim.unroll < dim.upcast) {
		const s = dim.st.shape.slice(dim.unroll, dim.upcast);
		addIndices(s, AluVar.unroll);
	}
	if (dim.upcast < dim.end) {
		const s = dim.st.shape.slice(dim.upcast);
		addIndices(s, AluVar.upcast);
	}
	let newExp = exp.rewrite((exp$1) => {
		if (exp$1.op === AluOp.GlobalView) {
			const gid = exp$1.arg[0];
			const st = exp$1.arg[1];
			return accessorGlobal(exp$1.dtype, gid, st.compose(dim.st), indices);
		}
	});
	const [iexpr, vexpr] = dim.st.toAluExp(indices);
	if (vexpr.min !== 1) throw new Error("Invariant violation: vexpr !== true");
	newExp = newExp.substitute({
		gidx: AluExp.idiv(iexpr, AluExp.i32(reduction.size)).simplify(),
		ridx: AluExp.mod(iexpr, AluExp.i32(reduction.size)).simplify()
	});
	const outputGidx = dim.outputSt.shape.slice(0, dim.groups);
	const outputUpcast = dim.outputSt.shape.slice(dim.groups);
	const outputIndices = [...unravelAlu(outputGidx, AluExp.special(DType.Int32, "gidx", prod(outputGidx))), ...unravelAlu(outputUpcast, AluVar.upcast)];
	const [outputIdxExp, _] = dim.outputSt.toAluExp(outputIndices);
	const newEpilogue = reduction.epilogue.rewrite((exp$1) => {
		if (exp$1.op === AluOp.GlobalView) {
			const gid = exp$1.arg[0];
			const st = exp$1.arg[1];
			return accessorGlobal(exp$1.dtype, gid, st.compose(dim.outputSt), outputIndices);
		}
	});
	if (prod(dim.st.shape.slice(dim.groups, dim.upcast)) !== reduction.size) throw new Error(`Invariant violation: reduction size ${reduction.size} does not match tuned dims ${JSON.stringify(dim.st.shape.slice(dim.groups, dim.upcast))}`);
	const size = {
		groups: prod(dim.st.shape.slice(dim.groups, dim.reduce)),
		reduce: prod(dim.st.shape.slice(dim.reduce, dim.unroll)),
		unroll: prod(dim.st.shape.slice(dim.unroll, dim.upcast)),
		upcast: prod(dim.st.shape.slice(dim.upcast))
	};
	return {
		exp: newExp.simplify(),
		epilogue: newEpilogue.simplify(),
		outputIdxExp: outputIdxExp.simplify(),
		threadCount: kernel.size / size.upcast * size.groups,
		size
	};
}

//#endregion
//#region src/backend/cpu.ts
/** Most basic implementation of `Backend` for testing. */
var CpuBackend = class {
	type = "cpu";
	maxArgs = Infinity;
	#buffers;
	#nextSlot;
	constructor() {
		this.#buffers = /* @__PURE__ */ new Map();
		this.#nextSlot = 1;
	}
	malloc(size, initialData) {
		const buffer = new Uint8Array(size);
		if (initialData) {
			if (initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
			buffer.set(initialData);
		}
		const slot = this.#nextSlot++;
		this.#buffers.set(slot, {
			buffer,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) this.#buffers.delete(slot);
	}
	slotCount() {
		return this.#buffers.size;
	}
	async read(slot, start, count) {
		return this.readSync(slot, start, count);
	}
	readSync(slot, start, count) {
		const buffer = this.#getBuffer(slot);
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.byteLength - start;
		return buffer.slice(start, start + count);
	}
	async prepareKernel(kernel) {
		return this.prepareKernelSync(kernel);
	}
	prepareKernelSync(kernel) {
		return new Executable(kernel, void 0);
	}
	async prepareRoutine(routine) {
		return this.prepareRoutineSync(routine);
	}
	prepareRoutineSync(routine) {
		return new Executable(routine, void 0);
	}
	dispatch(exe, inputs, outputs) {
		if (exe.source instanceof Routine) return runCpuRoutine(exe.source, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
		const kernel = exe.source;
		const { exp, epilogue } = tuneNullopt(kernel);
		const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
		const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
		const usedArgs = new Map([...exp.collect((exp$1) => exp$1.op === AluOp.GlobalIndex || exp$1.op === AluOp.GlobalView), ...epilogue ? epilogue.collect((exp$1) => exp$1.op === AluOp.GlobalIndex || exp$1.op === AluOp.GlobalView) : []].map((exp$1) => [exp$1.arg[0], exp$1.dtype]));
		const inputArrays = inputBuffers.map((buf, i) => {
			const dtype = usedArgs.get(i);
			if (!dtype) return null;
			return dtypedArray(dtype, buf);
		});
		const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);
		const globals = (gid, bufidx) => {
			if (gid < 0 || gid >= inputArrays.length) throw new Error("gid out of bounds: " + gid);
			if (bufidx < 0 || bufidx >= inputArrays[gid].length) throw new Error("bufidx out of bounds: " + bufidx);
			return inputArrays[gid][bufidx];
		};
		if (!kernel.reduction) for (let i = 0; i < kernel.size; i++) outputArray[i] = exp.evaluate({ gidx: i }, globals);
		else for (let i = 0; i < kernel.size; i++) {
			let acc = kernel.reduction.identity;
			for (let j = 0; j < kernel.reduction.size; j++) {
				const item = exp.evaluate({
					gidx: i,
					ridx: j
				}, globals);
				acc = kernel.reduction.evaluate(acc, item);
			}
			outputArray[i] = epilogue.evaluate({
				acc,
				gidx: i
			}, globals);
		}
	}
	#getBuffer(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		return buffer.buffer;
	}
};

//#endregion
//#region src/backend/wasm/allocator.ts
/** Simple tensor memory allocator for WebAssembly linear memory. */
var WasmAllocator = class {
	#memory;
	#headPtr;
	#freeLists;
	#allocatedBuffers;
	constructor(memory) {
		this.#memory = memory;
		this.#headPtr = 64;
		this.#freeLists = /* @__PURE__ */ new Map();
		this.#allocatedBuffers = /* @__PURE__ */ new Map();
	}
	malloc(size) {
		if (size === 0) return 0;
		const sizeClass = this.#findSizeClass(size);
		const freeList = this.#freeLists.get(sizeClass);
		let ptr;
		if (freeList && freeList.length > 0) {
			ptr = freeList.pop();
			new Uint8Array(this.#memory.buffer, ptr, sizeClass).fill(0);
		} else ptr = this.#bumpAlloc(sizeClass);
		this.#allocatedBuffers.set(ptr, sizeClass);
		return ptr;
	}
	free(ptr) {
		if (ptr === 0) return;
		const sizeClass = this.#allocatedBuffers.get(ptr);
		if (sizeClass === void 0) throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
		const freeList = this.#freeLists.get(sizeClass);
		if (freeList) freeList.push(ptr);
		else this.#freeLists.set(sizeClass, [ptr]);
		this.#allocatedBuffers.delete(ptr);
	}
	#bumpAlloc(size) {
		const ptr = this.#headPtr;
		size = size + 63 & -64;
		this.#headPtr += size;
		if (ptr + size > this.#memory.buffer.byteLength) this.#memory.grow((ptr + size + 65535 >> 16) - (this.#memory.buffer.byteLength >> 16));
		return ptr;
	}
	#findSizeClass(size) {
		if (size <= 512) return size + 63 & -64;
		if (size <= 2048) return size + 511 & -512;
		if (size <= 65536) {
			let sizeClass = 4096;
			while (sizeClass < size) sizeClass *= 2;
			return sizeClass;
		}
		return size + 65535 & -65536;
	}
	getStats() {
		const freeListSizes = /* @__PURE__ */ new Map();
		for (const [sizeClass, freeList] of this.#freeLists) if (freeList.length > 0) freeListSizes.set(sizeClass, freeList.length);
		return {
			totalAllocated: this.#headPtr,
			freeListSizes
		};
	}
};

//#endregion
//#region src/backend/wasm/wasmblr.ts
/**
* @file Minimalist WebAssembly assembler. This allows you to emit WebAssembly
* bytecode directly from the browser.
*
* Self-contained port of https://github.com/bwasti/wasmblr to TypeScript.
* Some operation names in this module are written in `snake_case` to match
* their names in the Wasm specification.
*
* Reference: https://pengowray.github.io/wasm-ops/.
*/
const magicModuleHeader = [
	0,
	97,
	115,
	109
];
const moduleVersion = [
	1,
	0,
	0,
	0
];
function assert(condition, message) {
	if (!condition) throw new Error(message || "Assertion failed");
}
function encodeSigned(n) {
	const out = [];
	let more = true;
	while (more) {
		let byte = n & 127;
		n >>= 7;
		if (n === 0 && (byte & 64) === 0 || n === -1 && (byte & 64) !== 0) more = false;
		else byte |= 128;
		out.push(byte);
	}
	return out;
}
function encodeUnsigned(n) {
	const out = [];
	do {
		let byte = n & 127;
		n = n >>> 7;
		if (n !== 0) byte |= 128;
		out.push(byte);
	} while (n !== 0);
	return out;
}
function encodeString(s) {
	const bytes = new TextEncoder().encode(s);
	return [bytes.length, ...bytes];
}
function encodeBlocktype(type) {
	assert(type.length > 0, "blocktype must have at least one type");
	if (type.length === 1) return [type[0].typeId];
	return [
		96,
		...encodeUnsigned(0),
		...encodeUnsigned(type.length),
		...type.map((t) => t.typeId)
	];
}
function encodeOpcode(opcode) {
	if (typeof opcode === "number") return [opcode];
	return [opcode[0], ...encodeUnsigned(opcode[1])];
}
function concat(out, inp) {
	out.push(...inp);
}
var Function_ = class {
	inputTypes;
	outputTypes;
	body;
	locals = [];
	constructor(inputTypes, outputTypes, body) {
		this.inputTypes = inputTypes;
		this.outputTypes = outputTypes;
		this.body = body || (() => {});
	}
	emit() {
		this.locals = [];
		this.body();
	}
};
var Memory = class {
	min = 0;
	max = 0;
	isShared = false;
	aString = "";
	bString = "";
	constructor(cg) {
		this.cg = cg;
	}
	/** Declare the size of the memory. Each page is 64 KiB. */
	pages(min, max = 0) {
		assert(this.min === 0 && this.max === 0);
		this.min = min;
		this.max = max;
		return this;
	}
	export(a) {
		assert(!this.isImport && !this.isExport, "already set");
		this.aString = a;
		return this;
	}
	shared(isShared) {
		this.isShared = isShared;
		return this;
	}
	import(a, b) {
		assert(!this.isImport && !this.isExport, "already set");
		this.aString = a;
		this.bString = b;
		return this;
	}
	size() {
		this.cg._emit(63);
		this.cg._emit(0);
	}
	grow() {
		this.cg._emit(64);
		this.cg._emit(0);
	}
	get isImport() {
		return this.aString.length > 0 && this.bString.length > 0;
	}
	get isExport() {
		return this.aString.length > 0 && this.bString.length === 0;
	}
};
/** Public API of WebAssembly assembler. */
var CodeGenerator = class {
	local;
	i32;
	f32;
	f64;
	v128;
	i32x4;
	f32x4;
	memory;
	void = {
		typeId: 64,
		name: "void"
	};
	#functions = [];
	#importedFunctions = [];
	#exportedFunctions = /* @__PURE__ */ new Map();
	#curFunction = null;
	#curBytes = [];
	#typeStack = [];
	#blockFrames = [];
	constructor() {
		this.local = new Local(this);
		this.i32 = new I32(this);
		this.f32 = new F32(this);
		this.f64 = new F64(this);
		this.v128 = new V128(this);
		this.i32x4 = new I32x4(this);
		this.f32x4 = new F32x4(this);
		this.memory = new Memory(this);
	}
	unreachable() {
		this._emit(0);
	}
	nop() {
		this._emit(1);
	}
	block(...type) {
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(2);
		this._emit(encodeBlocktype(type));
	}
	loop(...type) {
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(3);
		this._emit(encodeBlocktype(type));
	}
	if(...type) {
		assert(this._pop().typeId === this.i32.typeId, "if_: expected i32");
		this.#blockFrames.push({
			idx: this.#typeStack.length,
			ty: type
		});
		this._emit(4);
		this._emit(encodeBlocktype(type));
	}
	else() {
		assert(this.#blockFrames.length > 0, "else: no block to else");
		const frame = this.#blockFrames[this.#blockFrames.length - 1];
		this.#typeStack = this.#typeStack.slice(0, frame.idx);
		this._emit(5);
	}
	/** End a block (`block`, `if`/`else`, `loop`, or function). */
	end() {
		const frame = this.#blockFrames.pop();
		assert(frame !== void 0, "end: no block to end");
		this.#typeStack = this.#typeStack.slice(0, frame.idx);
		for (const ty of frame.ty) if (ty.typeId !== this.void.typeId) this._push(ty);
		this._emit(11);
	}
	/** Branch to a block a certain depth outward on the stack. */
	br(depth) {
		this._emit(12);
		this._emit(encodeUnsigned(depth));
	}
	/** Conditional branch to a block a certain depth outward on the stack. */
	br_if(depth) {
		assert(this._pop().typeId === this.i32.typeId, "br_if: expected i32");
		this._emit(13);
		this._emit(encodeUnsigned(depth));
	}
	/** Jump table that indexes into a label vector (like switch). */
	br_table(...depths) {
		assert(this._pop().typeId === this.i32.typeId, "br_table: expected i32");
		assert(depths.length > 0, "br_table: expected at least one default depth");
		this._emit(14);
		this._emit(encodeUnsigned(depths.length - 1));
		for (const d of depths) this._emit(encodeUnsigned(d));
	}
	/** Return from a function, branching out of the outermost block. */
	return() {
		this._emit(15);
	}
	/** Call a function with the given ID. */
	call(fn) {
		const totalFunctions = this.#importedFunctions.length + this.#functions.length;
		assert(fn < totalFunctions, "function index does not exist");
		const func = fn < this.#importedFunctions.length ? this.#importedFunctions[fn] : this.#functions[fn - this.#importedFunctions.length];
		for (let i = func.inputTypes.length - 1; i >= 0; i--) {
			const argType = this._pop();
			assert(argType.typeId === func.inputTypes[i].typeId, `call: argument ${i} type mismatch, expected ${func.inputTypes[i].name} got ${argType.name}`);
		}
		for (const outputType of func.outputTypes) this._push(outputType);
		this._emit(16);
		this._emit(encodeUnsigned(fn));
	}
	/** Throw away an operand on the stack. */
	drop() {
		this._pop();
		this._emit(26);
	}
	/** Select one of the first two operands (T, F) based on the third operand (i32)'s value. */
	select() {
		assert(this._pop().typeId === this.i32.typeId, "select: expected i32 condition");
		const [b, a] = [this._pop(), this._pop()];
		assert(a.typeId === b.typeId, "select: expected same type for both operands");
		this._push(a);
		this._emit(27);
	}
	/** Import a JavaScript function; returns its index. */
	importFunction(module, name, inputTypes, outputTypes) {
		if (this.#functions.length > 0) throw new Error("function imports must precede defining functions");
		const idx = this.#importedFunctions.length;
		this.#importedFunctions.push({
			module,
			name,
			inputTypes,
			outputTypes
		});
		return idx;
	}
	/** Export a function. */
	export(fn, name) {
		this.#exportedFunctions.set(fn, name);
	}
	/** Declare a new function; returns its index. */
	function(inputTypes, outputTypes, body) {
		const idx = this.#importedFunctions.length + this.#functions.length;
		this.#functions.push(new Function_(inputTypes, outputTypes, body));
		return idx;
	}
	_declareLocal(type) {
		assert(this.#curFunction !== null, "No current function");
		const idx = this.#curFunction.locals.length + this.#curFunction.inputTypes.length;
		this.#curFunction.locals.push(type);
		return idx;
	}
	_inputTypes() {
		assert(this.#curFunction !== null, "No current function");
		return this.#curFunction.inputTypes;
	}
	_locals() {
		assert(this.#curFunction !== null, "No current function");
		return this.#curFunction.locals;
	}
	_push(type) {
		if (!type) throw new Error(`pushing type ${type}`);
		this.#typeStack.push(type);
	}
	_pop() {
		assert(this.#typeStack.length > 0, "popping empty stack");
		return this.#typeStack.pop();
	}
	_emit(bytes) {
		if (typeof bytes === "number") this.#curBytes.push(bytes);
		else this.#curBytes.push(...bytes);
	}
	finish() {
		this.#curBytes = [];
		const emittedBytes = [];
		concat(emittedBytes, magicModuleHeader);
		concat(emittedBytes, moduleVersion);
		const typeSectionBytes = [];
		const totalFunctionTypes = this.#importedFunctions.length + this.#functions.length;
		concat(typeSectionBytes, encodeUnsigned(totalFunctionTypes));
		for (const f of [...this.#importedFunctions, ...this.#functions]) {
			typeSectionBytes.push(96);
			concat(typeSectionBytes, encodeUnsigned(f.inputTypes.length));
			for (const t of f.inputTypes) typeSectionBytes.push(t.typeId);
			concat(typeSectionBytes, encodeUnsigned(f.outputTypes.length));
			for (const t of f.outputTypes) typeSectionBytes.push(t.typeId);
		}
		emittedBytes.push(1);
		concat(emittedBytes, encodeUnsigned(typeSectionBytes.length));
		concat(emittedBytes, typeSectionBytes);
		const importSectionBytes = [];
		const numImports = this.#importedFunctions.length + (this.memory.isImport ? 1 : 0);
		if (numImports > 0) {
			concat(importSectionBytes, encodeUnsigned(numImports));
			for (let i = 0; i < this.#importedFunctions.length; i++) {
				const f = this.#importedFunctions[i];
				concat(importSectionBytes, encodeString(f.module));
				concat(importSectionBytes, encodeString(f.name));
				importSectionBytes.push(0);
				concat(importSectionBytes, encodeUnsigned(i));
			}
			if (this.memory.isImport) {
				concat(importSectionBytes, encodeString(this.memory.aString));
				concat(importSectionBytes, encodeString(this.memory.bString));
				importSectionBytes.push(2);
				if (this.memory.min && this.memory.max) {
					if (this.memory.isShared) importSectionBytes.push(3);
					else importSectionBytes.push(1);
					concat(importSectionBytes, encodeUnsigned(this.memory.min));
					concat(importSectionBytes, encodeUnsigned(this.memory.max));
				} else {
					assert(!this.memory.isShared, "shared memory must have a max size");
					importSectionBytes.push(0);
					concat(importSectionBytes, encodeUnsigned(this.memory.min));
				}
			}
			emittedBytes.push(2);
			concat(emittedBytes, encodeUnsigned(importSectionBytes.length));
			concat(emittedBytes, importSectionBytes);
		}
		const functionSectionBytes = [];
		concat(functionSectionBytes, encodeUnsigned(this.#functions.length));
		for (let i = 0; i < this.#functions.length; i++) {
			const typeIndex = this.#importedFunctions.length + i;
			concat(functionSectionBytes, encodeUnsigned(typeIndex));
		}
		emittedBytes.push(3);
		concat(emittedBytes, encodeUnsigned(functionSectionBytes.length));
		concat(emittedBytes, functionSectionBytes);
		const memorySectionBytes = [];
		if (!this.memory.isImport && (this.memory.min || this.memory.max)) {
			memorySectionBytes.push(1);
			if (this.memory.min && this.memory.max) {
				if (this.memory.isShared) memorySectionBytes.push(3);
				else memorySectionBytes.push(1);
				concat(memorySectionBytes, encodeUnsigned(this.memory.min));
				concat(memorySectionBytes, encodeUnsigned(this.memory.max));
			} else {
				assert(!this.memory.isShared, "shared memory must have a max size");
				memorySectionBytes.push(0);
				concat(memorySectionBytes, encodeUnsigned(this.memory.min));
			}
			emittedBytes.push(5);
			concat(emittedBytes, encodeUnsigned(memorySectionBytes.length));
			concat(emittedBytes, memorySectionBytes);
		}
		const exportSectionBytes = [];
		const numExports = this.#exportedFunctions.size + (this.memory.isExport ? 1 : 0);
		concat(exportSectionBytes, encodeUnsigned(numExports));
		if (this.memory.isExport) {
			concat(exportSectionBytes, encodeString(this.memory.aString));
			exportSectionBytes.push(2);
			exportSectionBytes.push(0);
		}
		for (const [key, name] of this.#exportedFunctions.entries()) {
			concat(exportSectionBytes, encodeString(name));
			exportSectionBytes.push(0);
			concat(exportSectionBytes, encodeUnsigned(key));
		}
		emittedBytes.push(7);
		concat(emittedBytes, encodeUnsigned(exportSectionBytes.length));
		concat(emittedBytes, exportSectionBytes);
		const codeSectionBytes = [];
		concat(codeSectionBytes, encodeUnsigned(this.#functions.length));
		for (const f of this.#functions) {
			this.#typeStack = [];
			this.#blockFrames = [{
				idx: 0,
				ty: f.outputTypes
			}];
			this.#curFunction = f;
			this.#curBytes = [];
			f.emit();
			this.end();
			const bodyBytes = [...this.#curBytes];
			this.#curBytes = [];
			concat(this.#curBytes, encodeUnsigned(f.locals.length));
			for (const l of f.locals) {
				this._emit(1);
				this._emit(l.typeId);
			}
			const headerBytes = [...this.#curBytes];
			const fnSize = headerBytes.length + bodyBytes.length;
			concat(codeSectionBytes, encodeUnsigned(fnSize));
			concat(codeSectionBytes, headerBytes);
			concat(codeSectionBytes, bodyBytes);
		}
		this.#curFunction = null;
		emittedBytes.push(10);
		concat(emittedBytes, encodeUnsigned(codeSectionBytes.length));
		concat(emittedBytes, codeSectionBytes);
		return new Uint8Array(emittedBytes);
	}
};
var Local = class {
	constructor(cg) {
		this.cg = cg;
	}
	declare(type) {
		return this.cg._declareLocal(type);
	}
	get(idx) {
		assert(Number.isInteger(idx), "getting non-integer local");
		const inputTypes = this.cg._inputTypes();
		if (idx < inputTypes.length) this.cg._push(inputTypes[idx]);
		else this.cg._push(this.cg._locals()[idx - inputTypes.length]);
		this.cg._emit(32);
		this.cg._emit(encodeUnsigned(idx));
	}
	set(idx) {
		const t = this.cg._pop();
		const inputTypes = this.cg._inputTypes();
		const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
		assert(expectedType.typeId === t.typeId, "can't set local to this value (wrong type)");
		this.cg._emit(33);
		this.cg._emit(encodeUnsigned(idx));
	}
	tee(idx) {
		const t = this.cg._pop();
		const inputTypes = this.cg._inputTypes();
		const expectedType = idx < inputTypes.length ? inputTypes[idx] : this.cg._locals()[idx - inputTypes.length];
		assert(expectedType.typeId === t.typeId, "can't tee local to this value (wrong type)");
		this.cg._emit(34);
		this.cg._emit(encodeUnsigned(idx));
		this.cg._push(expectedType);
	}
};
function UNARY_OP(op, opcode, inType, outType) {
	return function() {
		const t = this.cg._pop();
		assert(t.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inType} -> ${outType})`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._push(this.cg[outType]);
	};
}
function BINARY_OP(op, opcode, typeA, typeB, outType) {
	return function() {
		const b = this.cg._pop();
		const a = this.cg._pop();
		assert(a.typeId === this.cg[typeA].typeId && b.typeId === this.cg[typeB].typeId, `invalid type for ${op} (${typeA}, ${typeB} -> ${outType})`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._push(this.cg[outType]);
	};
}
function LOAD_OP(op, opcode, outType) {
	return function(align = 0, offset = 0) {
		const idxType = this.cg._pop();
		assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
		this.cg._push(this.cg[outType]);
	};
}
function STORE_OP(op, opcode, inType) {
	return function(align = 0, offset = 0) {
		const valType = this.cg._pop();
		const idxType = this.cg._pop();
		assert(valType.typeId === this.cg[inType].typeId, `invalid value type for ${op} (${inType})`);
		assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode(opcode));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
	};
}
var I32 = class {
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 127;
	}
	get name() {
		return "i32";
	}
	const(i) {
		this.cg._emit(65);
		this.cg._emit(encodeSigned(i));
		this.cg._push(this);
	}
	clz = UNARY_OP("clz", 103, "i32", "i32");
	ctz = UNARY_OP("ctz", 104, "i32", "i32");
	popcnt = UNARY_OP("popcnt", 105, "i32", "i32");
	lt_s = BINARY_OP("lt_s", 72, "i32", "i32", "i32");
	lt_u = BINARY_OP("lt_u", 73, "i32", "i32", "i32");
	gt_s = BINARY_OP("gt_s", 74, "i32", "i32", "i32");
	gt_u = BINARY_OP("gt_u", 75, "i32", "i32", "i32");
	le_s = BINARY_OP("le_s", 76, "i32", "i32", "i32");
	le_u = BINARY_OP("le_u", 77, "i32", "i32", "i32");
	ge_s = BINARY_OP("ge_s", 78, "i32", "i32", "i32");
	ge_u = BINARY_OP("ge_u", 79, "i32", "i32", "i32");
	add = BINARY_OP("add", 106, "i32", "i32", "i32");
	sub = BINARY_OP("sub", 107, "i32", "i32", "i32");
	mul = BINARY_OP("mul", 108, "i32", "i32", "i32");
	div_s = BINARY_OP("div_s", 109, "i32", "i32", "i32");
	div_u = BINARY_OP("div_u", 110, "i32", "i32", "i32");
	rem_s = BINARY_OP("rem_s", 111, "i32", "i32", "i32");
	rem_u = BINARY_OP("rem_u", 112, "i32", "i32", "i32");
	and = BINARY_OP("and", 113, "i32", "i32", "i32");
	or = BINARY_OP("or", 114, "i32", "i32", "i32");
	xor = BINARY_OP("xor", 115, "i32", "i32", "i32");
	shl = BINARY_OP("shl", 116, "i32", "i32", "i32");
	shr_s = BINARY_OP("shr_s", 117, "i32", "i32", "i32");
	shr_u = BINARY_OP("shr_u", 118, "i32", "i32", "i32");
	rotl = BINARY_OP("rotl", 119, "i32", "i32", "i32");
	rotr = BINARY_OP("rotr", 120, "i32", "i32", "i32");
	eqz = UNARY_OP("eqz", 69, "i32", "i32");
	eq = BINARY_OP("eq", 70, "i32", "i32", "i32");
	ne = BINARY_OP("ne", 71, "i32", "i32", "i32");
	trunc_f32_s = UNARY_OP("trunc_f32_s", 168, "f32", "i32");
	trunc_f32_u = UNARY_OP("trunc_f32_u", 169, "f32", "i32");
	trunc_f64_s = UNARY_OP("trunc_f64_s", 170, "f64", "i32");
	trunc_f64_u = UNARY_OP("trunc_f64_u", 171, "f64", "i32");
	load = LOAD_OP("load", 40, "i32");
	load8_s = LOAD_OP("load8_s", 44, "i32");
	load8_u = LOAD_OP("load8_u", 45, "i32");
	load16_s = LOAD_OP("load16_s", 46, "i32");
	load16_u = LOAD_OP("load16_u", 47, "i32");
	store = STORE_OP("store", 54, "i32");
	store8 = STORE_OP("store8", 58, "i32");
	store16 = STORE_OP("store16", 59, "i32");
	reinterpret_f32 = UNARY_OP("reinterpret_f32", 188, "f32", "i32");
	trunc_sat_f32_s = UNARY_OP("trunc_sat_f32_s", [252, 0], "f32", "i32");
	trunc_sat_f32_u = UNARY_OP("trunc_sat_f32_u", [252, 1], "f32", "i32");
	trunc_sat_f64_s = UNARY_OP("trunc_sat_f64_s", [252, 2], "f64", "i32");
	trunc_sat_f64_u = UNARY_OP("trunc_sat_f64_u", [252, 3], "f64", "i32");
};
var F32 = class {
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 125;
	}
	get name() {
		return "f32";
	}
	const(f) {
		this.cg._emit(67);
		const buffer = /* @__PURE__ */ new ArrayBuffer(4);
		new DataView(buffer).setFloat32(0, f, true);
		const bytes = new Uint8Array(buffer);
		for (let i = 0; i < 4; i++) this.cg._emit(bytes[i]);
		this.cg._push(this);
	}
	load = LOAD_OP("load", 42, "f32");
	store = STORE_OP("store", 56, "f32");
	eq = BINARY_OP("eq", 91, "f32", "f32", "i32");
	ne = BINARY_OP("ne", 92, "f32", "f32", "i32");
	lt = BINARY_OP("lt", 93, "f32", "f32", "i32");
	gt = BINARY_OP("gt", 94, "f32", "f32", "i32");
	le = BINARY_OP("le", 95, "f32", "f32", "i32");
	ge = BINARY_OP("ge", 96, "f32", "f32", "i32");
	abs = UNARY_OP("abs", 139, "f32", "f32");
	neg = UNARY_OP("neg", 140, "f32", "f32");
	ceil = UNARY_OP("ceil", 141, "f32", "f32");
	floor = UNARY_OP("floor", 142, "f32", "f32");
	trunc = UNARY_OP("trunc", 143, "f32", "f32");
	nearest = UNARY_OP("nearest", 144, "f32", "f32");
	sqrt = UNARY_OP("sqrt", 145, "f32", "f32");
	add = BINARY_OP("add", 146, "f32", "f32", "f32");
	sub = BINARY_OP("sub", 147, "f32", "f32", "f32");
	mul = BINARY_OP("mul", 148, "f32", "f32", "f32");
	div = BINARY_OP("div", 149, "f32", "f32", "f32");
	min = BINARY_OP("min", 150, "f32", "f32", "f32");
	max = BINARY_OP("max", 151, "f32", "f32", "f32");
	copysign = BINARY_OP("copysign", 152, "f32", "f32", "f32");
	convert_i32_s = UNARY_OP("convert_i32_s", 178, "i32", "f32");
	convert_i32_u = UNARY_OP("convert_i32_u", 179, "i32", "f32");
	demote_f64 = UNARY_OP("demote_f64", 182, "f64", "f32");
	reinterpret_i32 = UNARY_OP("reinterpret_i32", 190, "i32", "f32");
};
var F64 = class {
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 124;
	}
	get name() {
		return "f64";
	}
	const(f) {
		this.cg._emit(68);
		const buffer = /* @__PURE__ */ new ArrayBuffer(8);
		new DataView(buffer).setFloat64(0, f, true);
		const bytes = new Uint8Array(buffer);
		for (let i = 0; i < 8; i++) this.cg._emit(bytes[i]);
		this.cg._push(this);
	}
	load = LOAD_OP("load", 43, "f64");
	store = STORE_OP("store", 57, "f64");
	eq = BINARY_OP("eq", 97, "f64", "f64", "i32");
	ne = BINARY_OP("ne", 98, "f64", "f64", "i32");
	lt = BINARY_OP("lt", 99, "f64", "f64", "i32");
	gt = BINARY_OP("gt", 100, "f64", "f64", "i32");
	le = BINARY_OP("le", 101, "f64", "f64", "i32");
	ge = BINARY_OP("ge", 102, "f64", "f64", "i32");
	abs = UNARY_OP("abs", 153, "f64", "f64");
	neg = UNARY_OP("neg", 154, "f64", "f64");
	ceil = UNARY_OP("ceil", 155, "f64", "f64");
	floor = UNARY_OP("floor", 156, "f64", "f64");
	trunc = UNARY_OP("trunc", 157, "f64", "f64");
	nearest = UNARY_OP("nearest", 158, "f64", "f64");
	sqrt = UNARY_OP("sqrt", 159, "f64", "f64");
	add = BINARY_OP("add", 160, "f64", "f64", "f64");
	sub = BINARY_OP("sub", 161, "f64", "f64", "f64");
	mul = BINARY_OP("mul", 162, "f64", "f64", "f64");
	div = BINARY_OP("div", 163, "f64", "f64", "f64");
	min = BINARY_OP("min", 164, "f64", "f64", "f64");
	max = BINARY_OP("max", 165, "f64", "f64", "f64");
	copysign = BINARY_OP("copysign", 166, "f64", "f64", "f64");
	convert_i32_s = UNARY_OP("convert_i32_s", 183, "i32", "f64");
	convert_i32_u = UNARY_OP("convert_i32_u", 184, "i32", "f64");
	promote_f32 = UNARY_OP("promote_f32", 187, "f32", "f64");
};
function VECTOR_OP(op, vopcode, inTypes, outType) {
	return function() {
		for (const inType of inTypes.toReversed()) {
			const actualType = this.cg._pop();
			assert(actualType.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes.join(", ")} -> ${outType})`);
		}
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._push(this.cg[outType]);
	};
}
function VECTOR_OPL(op, vopcode, inTypes, outType) {
	return function(lane) {
		for (const inType of inTypes.toReversed()) {
			const actualType = this.cg._pop();
			assert(actualType.typeId === this.cg[inType].typeId, `invalid type for ${op} (${inTypes} -> ${outType})`);
		}
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._emit(lane);
		this.cg._push(this.cg[outType]);
	};
}
function VECTOR_LOAD_OP(op, vopcode) {
	return function(align = 0, offset = 0) {
		const idxType = this.cg._pop();
		assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
		this.cg._emit(encodeOpcode([253, vopcode]));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
		this.cg._push(this.cg.v128);
	};
}
var V128 = class {
	constructor(cg) {
		this.cg = cg;
	}
	get typeId() {
		return 123;
	}
	get name() {
		return "v128";
	}
	load = VECTOR_LOAD_OP("load", 0);
	load32x2_s = VECTOR_LOAD_OP("load32x2_s", 5);
	load32x2_u = VECTOR_LOAD_OP("load32x2_u", 6);
	load32_splat = VECTOR_LOAD_OP("load32_splat", 9);
	load32_zero = VECTOR_LOAD_OP("load32_zero", 92);
	store(align = 0, offset = 0) {
		const valType = this.cg._pop();
		assert(valType.typeId === this.cg.v128.typeId, `invalid type for store`);
		const idxType = this.cg._pop();
		assert(idxType.typeId === this.cg.i32.typeId, `invalid type for store`);
		this.cg._emit(253);
		this.cg._emit(encodeUnsigned(11));
		this.cg._emit(encodeUnsigned(align));
		this.cg._emit(encodeUnsigned(offset));
	}
	not = VECTOR_OP("not", 77, ["v128"], "v128");
	and = VECTOR_OP("and", 78, ["v128", "v128"], "v128");
	andnot = VECTOR_OP("andnot", 79, ["v128", "v128"], "v128");
	or = VECTOR_OP("or", 80, ["v128", "v128"], "v128");
	xor = VECTOR_OP("xor", 81, ["v128", "v128"], "v128");
	bitselect = VECTOR_OP("bitselect", 82, [
		"v128",
		"v128",
		"v128"
	], "v128");
	any_true = VECTOR_OP("any_true", 83, ["v128"], "i32");
};
var I32x4 = class extends V128 {
	splat = VECTOR_OP("splat", 17, ["i32"], "v128");
	extract_lane = VECTOR_OPL("extract_lane", 27, ["v128"], "i32");
	replace_lane = VECTOR_OPL("replace_lane", 28, ["v128", "i32"], "v128");
	eq = VECTOR_OP("eq", 55, ["v128", "v128"], "v128");
	ne = VECTOR_OP("ne", 56, ["v128", "v128"], "v128");
	lt_s = VECTOR_OP("lt_s", 57, ["v128", "v128"], "v128");
	lt_u = VECTOR_OP("lt_u", 58, ["v128", "v128"], "v128");
	gt_s = VECTOR_OP("gt_s", 59, ["v128", "v128"], "v128");
	gt_u = VECTOR_OP("gt_u", 60, ["v128", "v128"], "v128");
	le_s = VECTOR_OP("le_s", 61, ["v128", "v128"], "v128");
	le_u = VECTOR_OP("le_u", 62, ["v128", "v128"], "v128");
	ge_s = VECTOR_OP("ge_s", 63, ["v128", "v128"], "v128");
	ge_u = VECTOR_OP("ge_u", 64, ["v128", "v128"], "v128");
	abs = VECTOR_OP("abs", 160, ["v128"], "v128");
	neg = VECTOR_OP("neg", 161, ["v128"], "v128");
	all_true = VECTOR_OP("all_true", 163, ["v128"], "i32");
	bitmask = VECTOR_OP("bitmask", 164, ["v128"], "i32");
	shl = VECTOR_OP("shl", 171, ["v128", "i32"], "v128");
	shr_s = VECTOR_OP("shr_s", 172, ["v128", "i32"], "v128");
	shr_u = VECTOR_OP("shr_u", 173, ["v128", "i32"], "v128");
	add = VECTOR_OP("add", 174, ["v128", "v128"], "v128");
	sub = VECTOR_OP("sub", 177, ["v128", "v128"], "v128");
	mul = VECTOR_OP("mul", 181, ["v128", "v128"], "v128");
	min_s = VECTOR_OP("min_s", 182, ["v128", "v128"], "v128");
	min_u = VECTOR_OP("min_u", 183, ["v128", "v128"], "v128");
	max_s = VECTOR_OP("max_s", 184, ["v128", "v128"], "v128");
	max_u = VECTOR_OP("max_u", 185, ["v128", "v128"], "v128");
};
var F32x4 = class extends V128 {
	splat = VECTOR_OP("splat", 19, ["f32"], "v128");
	extract_lane = VECTOR_OPL("extract_lane", 31, ["v128"], "f32");
	replace_lane = VECTOR_OPL("replace_lane", 32, ["v128", "f32"], "v128");
	eq = VECTOR_OP("eq", 65, ["v128", "v128"], "v128");
	ne = VECTOR_OP("ne", 66, ["v128", "v128"], "v128");
	lt = VECTOR_OP("lt", 67, ["v128", "v128"], "v128");
	gt = VECTOR_OP("gt", 68, ["v128", "v128"], "v128");
	le = VECTOR_OP("le", 69, ["v128", "v128"], "v128");
	ge = VECTOR_OP("ge", 70, ["v128", "v128"], "v128");
	ceil = VECTOR_OP("ceil", 103, ["v128"], "v128");
	floor = VECTOR_OP("floor", 104, ["v128"], "v128");
	trunc = VECTOR_OP("trunc", 105, ["v128"], "v128");
	nearest = VECTOR_OP("nearest", 106, ["v128"], "v128");
	abs = VECTOR_OP("abs", 224, ["v128"], "v128");
	neg = VECTOR_OP("neg", 225, ["v128"], "v128");
	sqrt = VECTOR_OP("sqrt", 227, ["v128"], "v128");
	add = VECTOR_OP("add", 228, ["v128", "v128"], "v128");
	sub = VECTOR_OP("sub", 229, ["v128", "v128"], "v128");
	mul = VECTOR_OP("mul", 230, ["v128", "v128"], "v128");
	div = VECTOR_OP("div", 231, ["v128", "v128"], "v128");
	min = VECTOR_OP("min", 232, ["v128", "v128"], "v128");
	max = VECTOR_OP("max", 233, ["v128", "v128"], "v128");
	pmin = VECTOR_OP("pmin", 234, ["v128", "v128"], "v128");
	pmax = VECTOR_OP("pmax", 235, ["v128", "v128"], "v128");
};

//#endregion
//#region src/backend/wasm/argsort.ts
/**
* Generate a WASM function for argsort using merge sort.
*
* Sorts indices by comparing data values, producing both sorted data and indices.
*
* Function signature: (dataPtr, idxPtr, auxData, auxIdx, n) -> void
* - dataPtr: input/output data array [n]
* - idxPtr: output indices array [n] (int32)
* - auxData: auxiliary data buffer [n]
* - auxIdx: auxiliary index buffer [n] (int32)
* - n: array length
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_argsort(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	const idxSize = 4;
	const mergeFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const k = cg.local.declare(cg.i32);
		const vi = cg.local.declare(ft);
		const vj = cg.local.declare(ft);
		const ii = cg.local.declare(cg.i32);
		const ij = cg.local.declare(cg.i32);
		const srcData = 0;
		const srcIdx = 1;
		const dstData = 2;
		const dstIdx = 3;
		const left = 4;
		const mid = 5;
		const right = 6;
		cg.local.get(left);
		cg.local.set(i);
		cg.local.get(mid);
		cg.local.set(j);
		cg.local.get(left);
		cg.local.set(k);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(k);
		cg.local.get(right);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(i);
		cg.local.get(mid);
		cg.i32.ge_u();
		cg.if(cg.void);
		cg.local.get(dstData);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(srcData);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(dstIdx);
		cg.local.get(k);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(srcIdx);
		cg.local.get(j);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.i32.load(0, 0);
		cg.i32.store(0, 0);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.else();
		cg.local.get(j);
		cg.local.get(right);
		cg.i32.ge_u();
		cg.if(cg.void);
		cg.local.get(dstData);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(srcData);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(dstIdx);
		cg.local.get(k);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(srcIdx);
		cg.local.get(i);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.i32.load(0, 0);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.else();
		cg.local.get(srcData);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(vi);
		cg.local.get(srcData);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(vj);
		cg.local.get(srcIdx);
		cg.local.get(i);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.i32.load(0, 0);
		cg.local.set(ii);
		cg.local.get(srcIdx);
		cg.local.get(j);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.i32.load(0, 0);
		cg.local.set(ij);
		cg.local.get(vj);
		cg.local.get(vj);
		ft.ne();
		cg.local.get(vi);
		cg.local.get(vi);
		ft.eq();
		cg.local.get(vi);
		cg.local.get(vj);
		ft.lt();
		cg.i32.and();
		cg.i32.or();
		cg.if(cg.void);
		cg.local.get(dstData);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(vi);
		ft.store(0, 0);
		cg.local.get(dstIdx);
		cg.local.get(k);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(ii);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.else();
		cg.local.get(vi);
		cg.local.get(vj);
		ft.eq();
		cg.local.get(ii);
		cg.local.get(ij);
		cg.i32.lt_s();
		cg.i32.and();
		cg.if(cg.void);
		cg.local.get(dstData);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(vi);
		ft.store(0, 0);
		cg.local.get(dstIdx);
		cg.local.get(k);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(ii);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.else();
		cg.local.get(dstData);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(vj);
		ft.store(0, 0);
		cg.local.get(dstIdx);
		cg.local.get(k);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(ij);
		cg.i32.store(0, 0);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.end();
		cg.end();
		cg.end();
		cg.end();
		cg.local.get(k);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.br(1);
		cg.end();
		cg.end();
	});
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const width = cg.local.declare(cg.i32);
		const i = cg.local.declare(cg.i32);
		const left = cg.local.declare(cg.i32);
		const mid = cg.local.declare(cg.i32);
		const right = cg.local.declare(cg.i32);
		const srcData = cg.local.declare(cg.i32);
		const srcIdx = cg.local.declare(cg.i32);
		const dstData = cg.local.declare(cg.i32);
		const dstIdx = cg.local.declare(cg.i32);
		const pass = cg.local.declare(cg.i32);
		const dataPtr = 0;
		const idxPtr = 1;
		const auxData = 2;
		const auxIdx = 3;
		const n = 4;
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(idxPtr);
		cg.local.get(i);
		cg.i32.const(4);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(i);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.i32.const(1);
		cg.local.set(width);
		cg.i32.const(0);
		cg.local.set(pass);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(width);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.and();
		cg.i32.eqz();
		cg.if(cg.void);
		cg.local.get(dataPtr);
		cg.local.set(srcData);
		cg.local.get(idxPtr);
		cg.local.set(srcIdx);
		cg.local.get(auxData);
		cg.local.set(dstData);
		cg.local.get(auxIdx);
		cg.local.set(dstIdx);
		cg.else();
		cg.local.get(auxData);
		cg.local.set(srcData);
		cg.local.get(auxIdx);
		cg.local.set(srcIdx);
		cg.local.get(dataPtr);
		cg.local.set(dstData);
		cg.local.get(idxPtr);
		cg.local.set(dstIdx);
		cg.end();
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(i);
		cg.local.set(left);
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.add();
		cg.local.set(mid);
		cg.local.get(mid);
		cg.local.get(n);
		cg.i32.gt_u();
		cg.if(cg.void);
		cg.local.get(n);
		cg.local.set(mid);
		cg.end();
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(right);
		cg.local.get(right);
		cg.local.get(n);
		cg.i32.gt_u();
		cg.if(cg.void);
		cg.local.get(n);
		cg.local.set(right);
		cg.end();
		cg.local.get(srcData);
		cg.local.get(srcIdx);
		cg.local.get(dstData);
		cg.local.get(dstIdx);
		cg.local.get(left);
		cg.local.get(mid);
		cg.local.get(right);
		cg.call(mergeFn);
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.local.set(width);
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(pass);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.and();
		cg.if(cg.void);
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(dataPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(auxData);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(idxPtr);
		cg.local.get(i);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(auxIdx);
		cg.local.get(i);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.i32.add();
		cg.i32.load(0, 0);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM module for batched argsort.
*
* The exported function takes:
*   (inPtr, outDataPtr, outIdxPtr, auxData, auxIdx, n, batchSize)
*
* @param elementSize - Element size in bytes: 4 for f32, 8 for f64
* @returns The compiled WebAssembly.Module
*/
function createArgsortModule(elementSize = 4) {
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const argsortFn = wasm_argsort(cg, ft);
	const idxSize = 4;
	const batchedFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const batch = cg.local.declare(cg.i32);
		const dataArraySize = cg.local.declare(cg.i32);
		const idxArraySize = cg.local.declare(cg.i32);
		const inOffset = cg.local.declare(cg.i32);
		const outDataOffset = cg.local.declare(cg.i32);
		const outIdxOffset = cg.local.declare(cg.i32);
		const i = cg.local.declare(cg.i32);
		const inPtr = 0;
		const outDataPtr = 1;
		const outIdxPtr = 2;
		const auxData = 3;
		const auxIdx = 4;
		const n = 5;
		const batchSize = 6;
		cg.local.get(n);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(dataArraySize);
		cg.local.get(n);
		cg.i32.const(idxSize);
		cg.i32.mul();
		cg.local.set(idxArraySize);
		cg.i32.const(0);
		cg.local.set(batch);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(batch);
		cg.local.get(batchSize);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(inPtr);
		cg.local.get(batch);
		cg.local.get(dataArraySize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(inOffset);
		cg.local.get(outDataPtr);
		cg.local.get(batch);
		cg.local.get(dataArraySize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(outDataOffset);
		cg.local.get(outIdxPtr);
		cg.local.get(batch);
		cg.local.get(idxArraySize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(outIdxOffset);
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(outDataOffset);
		cg.local.get(i);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(inOffset);
		cg.local.get(i);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(outDataOffset);
		cg.local.get(outIdxOffset);
		cg.local.get(auxData);
		cg.local.get(auxIdx);
		cg.local.get(n);
		cg.call(argsortFn);
		cg.local.get(batch);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(batch);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(batchedFn, "argsort");
	return new WebAssembly.Module(cg.finish());
}

//#endregion
//#region src/backend/wasm/builtins.ts
/** Given a local `x`, evaluate `sum[i](a_i * x^i)` and push to stack. */
function _poly(cg, x, as) {
	if (as.length === 0) throw new Error("_poly needs at least one coefficient");
	cg.f32.const(as[as.length - 1]);
	for (let i = as.length - 2; i >= 0; i--) {
		cg.local.get(x);
		cg.f32.mul();
		if (as[i] !== 0) {
			cg.f32.const(as[i]);
			cg.f32.add();
		}
	}
}
/**
* Approximate e^x.
*
* Method: range-reduce x = k*ln2 + r with k = round(x/ln2), |r|<=~0.3466
*         then e^x = 2^k * P(r), where P is 5th-order poly (Taylor).
*/
function wasm_exp(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const k_f = cg.local.declare(cg.f32);
		const k = cg.local.declare(cg.i32);
		const r = cg.local.declare(cg.f32);
		const p = cg.local.declare(cg.f32);
		const scale = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.const(1 / Math.LN2);
		cg.f32.mul();
		cg.f32.nearest();
		cg.local.tee(k_f);
		cg.i32.trunc_sat_f32_s();
		cg.local.set(k);
		cg.local.get(k);
		cg.i32.const(127);
		cg.i32.gt_s();
		cg.if(cg.void);
		cg.f32.const(Infinity);
		cg.return();
		cg.end();
		cg.local.get(k);
		cg.i32.const(-126);
		cg.i32.lt_s();
		cg.if(cg.void);
		cg.f32.const(0);
		cg.return();
		cg.end();
		cg.local.get(0);
		cg.local.get(k_f);
		cg.f32.const(Math.LN2);
		cg.f32.mul();
		cg.f32.sub();
		cg.local.set(r);
		_poly(cg, r, [
			1,
			1,
			1 / 2,
			1 / 6,
			1 / 24,
			1 / 120,
			1 / 720
		]);
		cg.local.set(p);
		cg.local.get(k);
		cg.i32.const(127);
		cg.i32.add();
		cg.i32.const(23);
		cg.i32.shl();
		cg.f32.reinterpret_i32();
		cg.local.set(scale);
		cg.local.get(p);
		cg.local.get(scale);
		cg.f32.mul();
	});
}
/**
* Approximate ln(x), x > 0.
*
* Method: decompose x = m * 2^e with m in [1,2), e integer (via bit ops)
*         ln(x) = e*ln2 + ln(m);  use atanh-style series with t=(m-1)/(m+1)
*         ln(m) ≈ 2*(t + t^3/3 + t^5/5 + t^7/7)
*/
function wasm_log(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const bits = cg.local.declare(cg.i32);
		const e = cg.local.declare(cg.i32);
		const m = cg.local.declare(cg.f32);
		const t = cg.local.declare(cg.f32);
		const t2 = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.const(0);
		cg.f32.lt();
		cg.if(cg.void);
		cg.f32.const(NaN);
		cg.return();
		cg.end();
		cg.local.get(0);
		cg.i32.reinterpret_f32();
		cg.local.tee(bits);
		cg.i32.const(23);
		cg.i32.shr_u();
		cg.i32.const(255);
		cg.i32.and();
		cg.i32.const(127);
		cg.i32.sub();
		cg.local.set(e);
		cg.local.get(e);
		cg.i32.const(-127);
		cg.i32.eq();
		cg.if(cg.void);
		cg.f32.const(-Infinity);
		cg.return();
		cg.end();
		cg.local.get(e);
		cg.i32.const(128);
		cg.i32.eq();
		cg.if(cg.void);
		cg.local.get(0);
		cg.return();
		cg.end();
		cg.local.get(bits);
		cg.i32.const(8388607);
		cg.i32.and();
		cg.i32.const(1065353216);
		cg.i32.or();
		cg.f32.reinterpret_i32();
		cg.local.set(m);
		cg.local.get(m);
		cg.f32.const(1);
		cg.f32.sub();
		cg.local.get(m);
		cg.f32.const(1);
		cg.f32.add();
		cg.f32.div();
		cg.local.set(t);
		cg.local.get(t);
		cg.local.get(t);
		cg.f32.mul();
		cg.local.set(t2);
		_poly(cg, t2, [
			2,
			2 / 3,
			2 / 5,
			2 / 7
		]);
		cg.local.get(t);
		cg.f32.mul();
		cg.local.get(e);
		cg.f32.convert_i32_s();
		cg.f32.const(Math.LN2);
		cg.f32.mul();
		cg.f32.add();
	});
}
/**
* Common helper to approximate sin(x) and cos(x).
*
* Method: reduce to y in [-π, π], then quadrant via q = round(y/(π/2))
*         z = y - q*(π/2); use one of two polynomials on z:
*         sin(z) ≈ z + z^3*(-1/6) + z^5*(1/120) + z^7*(-1/5040)
*         cos(z) ≈ 1 + z^2*(-1/2) + z^4*(1/24) + z^6*(-1/720) + z^8*(1/40320)
*/
function _sincos(cg) {
	const y = cg.local.declare(cg.f32);
	const qf = cg.local.declare(cg.f32);
	const q = cg.local.declare(cg.i32);
	const z = cg.local.declare(cg.f32);
	const z2 = cg.local.declare(cg.f32);
	const sz = cg.local.declare(cg.f32);
	const cz = cg.local.declare(cg.f32);
	cg.local.get(0);
	cg.local.get(0);
	cg.f32.const(1 / (2 * Math.PI));
	cg.f32.mul();
	cg.f32.nearest();
	cg.local.tee(qf);
	cg.f32.const(2 * Math.PI);
	cg.f32.mul();
	cg.f32.sub();
	cg.local.set(y);
	cg.local.get(y);
	cg.f32.const(2 / Math.PI);
	cg.f32.mul();
	cg.f32.nearest();
	cg.local.tee(qf);
	cg.i32.trunc_sat_f32_s();
	cg.local.set(q);
	cg.local.get(y);
	cg.local.get(qf);
	cg.f32.const(Math.PI / 2);
	cg.f32.mul();
	cg.f32.sub();
	cg.local.tee(z);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(z2);
	_poly(cg, z2, [
		1,
		-1 / 6,
		1 / 120,
		-1 / 5040
	]);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(sz);
	_poly(cg, z2, [
		1,
		-1 / 2,
		1 / 24,
		-1 / 720,
		1 / 40320
	]);
	cg.local.set(cz);
	return {
		q,
		sz,
		cz
	};
}
/**
* Approximate sin(x).
*
* Quadrant mapping: k=q mod 4: 0: +sz, 1: +cz, 2: -sz, 3: -cz
*/
function wasm_sin(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const { q, sz, cz } = _sincos(cg);
		const mag = cg.local.declare(cg.f32);
		cg.local.get(cz);
		cg.local.get(sz);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.and();
		cg.select();
		cg.local.tee(mag);
		cg.f32.neg();
		cg.local.get(mag);
		cg.local.get(q);
		cg.i32.const(2);
		cg.i32.and();
		cg.select();
	});
}
/**
* Approximate cos(x).
*
* Quadrant mapping: k=q mod 4: 0: +cz, 1: -sz, 2: -cz, 3: +sz
*/
function wasm_cos(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		const { q, sz, cz } = _sincos(cg);
		const mag = cg.local.declare(cg.f32);
		cg.local.get(sz);
		cg.local.get(cz);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.and();
		cg.select();
		cg.local.tee(mag);
		cg.f32.neg();
		cg.local.get(mag);
		cg.local.get(q);
		cg.i32.const(1);
		cg.i32.add();
		cg.i32.const(2);
		cg.i32.and();
		cg.select();
	});
}
/** Helper function for approximating arctan(x).  */
function _atan(cg) {
	const x = cg.local.declare(cg.f32);
	const abs_x = cg.local.declare(cg.f32);
	const z = cg.local.declare(cg.f32);
	const z2 = cg.local.declare(cg.f32);
	const p = cg.local.declare(cg.f32);
	cg.local.set(x);
	cg.local.get(x);
	cg.f32.abs();
	cg.local.set(abs_x);
	cg.f32.const(1);
	cg.local.get(abs_x);
	cg.f32.div();
	cg.local.get(abs_x);
	cg.local.get(abs_x);
	cg.f32.const(1);
	cg.f32.ge();
	cg.select();
	cg.local.set(z);
	cg.local.get(z);
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(z2);
	_poly(cg, z2, [
		.999998614341,
		.661705427875,
		.0415796528637
	]);
	_poly(cg, z2, [
		1,
		.994987933645,
		.173698870181
	]);
	cg.f32.div();
	cg.local.get(z);
	cg.f32.mul();
	cg.local.set(p);
	cg.f32.const(Math.PI / 2);
	cg.local.get(p);
	cg.f32.sub();
	cg.local.get(p);
	cg.local.get(abs_x);
	cg.f32.const(1);
	cg.f32.ge();
	cg.select();
	cg.local.get(x);
	cg.f32.copysign();
}
/**
* Approximate atan(x).
*
* Method: if |x| < 1, use rational approximation: atan(x) ≈ x * P(x^2) / Q(x^2)
*         where P(u) = A0 + A1*u + A2*u^2 (degree 2)
*               Q(u) = 1 + B1*u + B2*u^2 (degree 2)
*         if |x| >= 1, use: atan(x) = sign(x)*π/2 - atan(1/x)
*         (fitted coefficients, max error ~5e-7 on [0,1])
*/
function wasm_atan(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.local.get(0);
		_atan(cg);
	});
}
/**
* Approximate asin(x).
*
* Method: asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
*/
function wasm_asin(cg) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.local.get(0);
		cg.f32.const(1);
		cg.local.get(0);
		cg.local.get(0);
		cg.f32.mul();
		cg.f32.sub();
		cg.f32.sqrt();
		cg.f32.const(1);
		cg.f32.add();
		cg.f32.div();
		_atan(cg);
		cg.f32.const(2);
		cg.f32.mul();
	});
}
/**
* Helper function for erf/erfc approximation.
*
* See `_erfapprox` in alu.ts for details on the algorithm used.
*/
function _erfapprox(cg, exp_func) {
	const x = cg.local.declare(cg.f32);
	const t = cg.local.declare(cg.f32);
	cg.local.set(x);
	const p = .3275911;
	const a1 = .254829592;
	const a2 = -.284496736;
	const a3 = 1.421413741;
	const a4 = -1.453152027;
	const a5 = 1.061405429;
	cg.f32.const(1);
	cg.f32.const(1);
	cg.f32.const(p);
	cg.local.get(x);
	cg.f32.mul();
	cg.f32.add();
	cg.f32.div();
	cg.local.set(t);
	_poly(cg, t, [
		0,
		a1,
		a2,
		a3,
		a4,
		a5
	]);
	cg.local.get(x);
	cg.f32.neg();
	cg.local.get(x);
	cg.f32.mul();
	cg.call(exp_func);
	cg.f32.mul();
}
/** Approximate erf(x) (error function). */
function wasm_erf(cg, exp) {
	return cg.function([cg.f32], [cg.f32], () => {
		cg.f32.const(1);
		cg.local.get(0);
		cg.f32.abs();
		_erfapprox(cg, exp);
		cg.f32.sub();
		cg.local.get(0);
		cg.f32.copysign();
	});
}
/** Approximate erfc(x) (complementary error function). */
function wasm_erfc(cg, exp) {
	return cg.function([cg.f32], [cg.f32], () => {
		const e = cg.local.declare(cg.f32);
		cg.local.get(0);
		cg.f32.abs();
		_erfapprox(cg, exp);
		cg.local.set(e);
		cg.f32.const(2);
		cg.local.get(e);
		cg.f32.sub();
		cg.local.get(e);
		cg.local.get(0);
		cg.f32.const(0);
		cg.f32.lt();
		cg.select();
	});
}
/**
* Threefry2x32 pseudorandom number generator.
*
* Takes two 32-bit keys and two 32-bit counters as input,
* returns two 32-bit pseudorandom values.
*/
function wasm_threefry2x32(cg) {
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [cg.i32, cg.i32], () => {
		const ks0 = cg.local.declare(cg.i32);
		const ks1 = cg.local.declare(cg.i32);
		const ks2 = cg.local.declare(cg.i32);
		const x0 = cg.local.declare(cg.i32);
		const x1 = cg.local.declare(cg.i32);
		const mix = (rot) => {
			cg.local.get(x0);
			cg.local.get(x1);
			cg.i32.add();
			cg.local.set(x0);
			cg.local.get(x1);
			cg.i32.const(rot);
			cg.i32.rotl();
			cg.local.get(x0);
			cg.i32.xor();
			cg.local.set(x1);
		};
		const keySchedule = (k0, k1, round) => {
			cg.local.get(x0);
			cg.local.get(k0);
			cg.i32.add();
			cg.local.set(x0);
			cg.local.get(x1);
			cg.local.get(k1);
			cg.i32.add();
			cg.i32.const(round);
			cg.i32.add();
			cg.local.set(x1);
		};
		cg.local.get(0);
		cg.local.set(ks0);
		cg.local.get(1);
		cg.local.set(ks1);
		cg.local.get(0);
		cg.local.get(1);
		cg.i32.xor();
		cg.i32.const(466688986);
		cg.i32.xor();
		cg.local.set(ks2);
		cg.local.get(2);
		cg.local.get(ks0);
		cg.i32.add();
		cg.local.set(x0);
		cg.local.get(3);
		cg.local.get(ks1);
		cg.i32.add();
		cg.local.set(x1);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks1, ks2, 1);
		mix(17), mix(29), mix(16), mix(24);
		keySchedule(ks2, ks0, 2);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks0, ks1, 3);
		mix(17), mix(29), mix(16), mix(24);
		keySchedule(ks1, ks2, 4);
		mix(13), mix(15), mix(26), mix(6);
		keySchedule(ks2, ks0, 5);
		cg.local.get(x0);
		cg.local.get(x1);
	});
}

//#endregion
//#region src/backend/wasm/cholesky.ts
/**
* Generate a WASM function for Cholesky decomposition.
*
* The function takes (inPtr: i32, outPtr: i32, n: i32) and computes
* the Cholesky decomposition of a single n×n matrix.
*
* Algorithm (Cholesky-Banachiewicz):
*   for i = 0 to n-1:
*     for j = 0 to i:
*       sum = A[i,j]
*       for k = 0 to j-1:
*         sum -= L[i,k] * L[j,k]
*       if i == j:
*         L[i,j] = sqrt(sum)
*       else:
*         L[i,j] = sum / L[j,j]
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_cholesky(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const k = cg.local.declare(cg.i32);
		const sum = cg.local.declare(ft);
		cg.local.declare(ft);
		const ijOffset = cg.local.declare(cg.i32);
		const jjOffset = cg.local.declare(cg.i32);
		const ikOffset = cg.local.declare(cg.i32);
		const jkOffset = cg.local.declare(cg.i32);
		const nBytes = cg.local.declare(cg.i32);
		const inPtr = 0;
		const outPtr = 1;
		const n = 2;
		cg.local.get(n);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.local.set(nBytes);
		const idx = cg.local.declare(cg.i32);
		const nn = cg.local.declare(cg.i32);
		cg.local.get(n);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.set(nn);
		cg.i32.const(0);
		cg.local.set(idx);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(idx);
		cg.local.get(nn);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(outPtr);
		cg.local.get(idx);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.const(0);
		ft.store(0, 0);
		cg.local.get(idx);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(idx);
		cg.br(1);
		cg.end();
		cg.end();
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.i32.const(0);
		cg.local.set(j);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(j);
		cg.local.get(i);
		cg.i32.gt_u();
		cg.br_if(0);
		cg.local.get(outPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(ijOffset);
		cg.local.get(inPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(sum);
		cg.i32.const(0);
		cg.local.set(k);
		cg.block(cg.void);
		cg.local.get(j);
		cg.i32.eqz();
		cg.br_if(0);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(k);
		cg.local.get(j);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(outPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(ikOffset);
		cg.local.get(outPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(jkOffset);
		cg.local.get(sum);
		cg.local.get(ikOffset);
		ft.load(0, 0);
		cg.local.get(jkOffset);
		ft.load(0, 0);
		ft.mul();
		ft.sub();
		cg.local.set(sum);
		cg.local.get(k);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.br(1);
		cg.end();
		cg.end();
		cg.end();
		cg.local.get(i);
		cg.local.get(j);
		cg.i32.eq();
		cg.if(cg.void);
		cg.local.get(ijOffset);
		cg.local.get(sum);
		ft.sqrt();
		ft.store(0, 0);
		cg.else();
		cg.local.get(outPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(jjOffset);
		cg.local.get(ijOffset);
		cg.local.get(sum);
		cg.local.get(jjOffset);
		ft.load(0, 0);
		ft.div();
		ft.store(0, 0);
		cg.end();
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM module for batched Cholesky decomposition.
*
* The exported function takes (inPtr, outPtr, n, batchSize) and computes
* Cholesky decomposition for each matrix in the batch.
*
* Creates either an f32 or f64 module based on elementSize.
*
* @param elementSize - Element size in bytes: 4 for f32, 8 for f64
* @returns The compiled WebAssembly.Module
*/
function createCholeskyModule(elementSize = 4) {
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const choleskyFn = wasm_cholesky(cg, ft);
	const batchedFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const batch = cg.local.declare(cg.i32);
		const matrixSize = cg.local.declare(cg.i32);
		const inOffset = cg.local.declare(cg.i32);
		const outOffset = cg.local.declare(cg.i32);
		cg.local.get(2);
		cg.local.get(2);
		cg.i32.mul();
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(matrixSize);
		cg.i32.const(0);
		cg.local.set(batch);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(batch);
		cg.local.get(3);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(0);
		cg.local.get(batch);
		cg.local.get(matrixSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(inOffset);
		cg.local.get(1);
		cg.local.get(batch);
		cg.local.get(matrixSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(outOffset);
		cg.local.get(inOffset);
		cg.local.get(outOffset);
		cg.local.get(2);
		cg.call(choleskyFn);
		cg.local.get(batch);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(batch);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(batchedFn, "cholesky");
	return new WebAssembly.Module(cg.finish());
}

//#endregion
//#region src/backend/wasm/lu.ts
/**
* Generate a WASM function for LU decomposition of a single matrix.
*
* Algorithm (Gaussian elimination with partial pivoting):
*   for j = 0 to min(m,n)-1:
*     // Find pivot (max absolute value in column j, rows j..m-1)
*     maxRow = j
*     maxVal = |LU[j,j]|
*     for i = j+1 to m-1:
*       if |LU[i,j]| > maxVal:
*         maxVal = |LU[i,j]|
*         maxRow = i
*     pivots[j] = maxRow
*
*     // Swap rows j and maxRow
*     if maxRow != j:
*       swap row j with row maxRow in LU
*       swap perm[j] with perm[maxRow]
*
*     // Eliminate below pivot
*     if LU[j,j] != 0:
*       for i = j+1 to m-1:
*         factor = LU[i,j] / LU[j,j]
*         LU[i,j] = factor  // Store L factor
*         for k = j+1 to n-1:
*           LU[i,k] -= factor * LU[j,k]
*
* Function signature:
*   (inPtr, luPtr, pivotsPtr, permPtr, m, n) -> void
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_lu(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const k = cg.local.declare(cg.i32);
		const r = cg.local.declare(cg.i32);
		const maxRow = cg.local.declare(cg.i32);
		const maxVal = cg.local.declare(ft);
		const curVal = cg.local.declare(ft);
		const diag = cg.local.declare(ft);
		const factor = cg.local.declare(ft);
		const tmp = cg.local.declare(ft);
		const tmpI = cg.local.declare(cg.i32);
		const offset1 = cg.local.declare(cg.i32);
		const offset2 = cg.local.declare(cg.i32);
		const inPtr = 0;
		const luPtr = 1;
		const pivotsPtr = 2;
		const permPtr = 3;
		const m = 4;
		const n = 5;
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(m);
		cg.local.get(n);
		cg.i32.mul();
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(luPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(inPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(m);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(permPtr);
		cg.local.get(i);
		cg.i32.const(4);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(i);
		cg.i32.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(m);
		cg.local.get(n);
		cg.i32.lt_u();
		cg.if(cg.i32);
		cg.local.get(m);
		cg.else();
		cg.local.get(n);
		cg.end();
		cg.local.set(r);
		cg.i32.const(0);
		cg.local.set(j);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(j);
		cg.local.get(r);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(j);
		cg.local.set(maxRow);
		cg.local.get(luPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.abs();
		cg.local.set(maxVal);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(m);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(luPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.abs();
		cg.local.set(curVal);
		cg.local.get(curVal);
		cg.local.get(maxVal);
		ft.gt();
		cg.if(cg.void);
		cg.local.get(curVal);
		cg.local.set(maxVal);
		cg.local.get(i);
		cg.local.set(maxRow);
		cg.end();
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(pivotsPtr);
		cg.local.get(j);
		cg.i32.const(4);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(maxRow);
		cg.i32.store(0, 0);
		cg.local.get(maxRow);
		cg.local.get(j);
		cg.i32.ne();
		cg.if(cg.void);
		cg.i32.const(0);
		cg.local.set(k);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(k);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(luPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset1);
		cg.local.get(luPtr);
		cg.local.get(maxRow);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset2);
		cg.local.get(offset1);
		ft.load(0, 0);
		cg.local.set(tmp);
		cg.local.get(offset1);
		cg.local.get(offset2);
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(offset2);
		cg.local.get(tmp);
		ft.store(0, 0);
		cg.local.get(k);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(permPtr);
		cg.local.get(j);
		cg.i32.const(4);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset1);
		cg.local.get(permPtr);
		cg.local.get(maxRow);
		cg.i32.const(4);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset2);
		cg.local.get(offset1);
		cg.i32.load(0, 0);
		cg.local.set(tmpI);
		cg.local.get(offset1);
		cg.local.get(offset2);
		cg.i32.load(0, 0);
		cg.i32.store(0, 0);
		cg.local.get(offset2);
		cg.local.get(tmpI);
		cg.i32.store(0, 0);
		cg.end();
		cg.local.get(luPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(diag);
		cg.local.get(diag);
		ft.const(0);
		ft.ne();
		cg.if(cg.void);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(m);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(luPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset1);
		cg.local.get(offset1);
		ft.load(0, 0);
		cg.local.get(diag);
		ft.div();
		cg.local.set(factor);
		cg.local.get(offset1);
		cg.local.get(factor);
		ft.store(0, 0);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(k);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(luPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset1);
		cg.local.get(luPtr);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(k);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(offset2);
		cg.local.get(offset1);
		cg.local.get(offset1);
		ft.load(0, 0);
		cg.local.get(factor);
		cg.local.get(offset2);
		ft.load(0, 0);
		ft.mul();
		ft.sub();
		ft.store(0, 0);
		cg.local.get(k);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.end();
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.br(1);
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM module for batched LU decomposition.
*
* The exported function takes:
*   (inPtr, luPtr, pivotsPtr, permPtr, m, n, batchSize)
*
* @param elementSize - Element size in bytes: 4 for f32, 8 for f64
* @returns The compiled WebAssembly.Module
*/
function createLUModule(elementSize = 4) {
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const luFn = wasm_lu(cg, ft);
	const batchedFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const batch = cg.local.declare(cg.i32);
		const matrixSize = cg.local.declare(cg.i32);
		const pivotSize = cg.local.declare(cg.i32);
		const permSize = cg.local.declare(cg.i32);
		const inOffset = cg.local.declare(cg.i32);
		const luOffset = cg.local.declare(cg.i32);
		const pivotsOffset = cg.local.declare(cg.i32);
		const permOffset = cg.local.declare(cg.i32);
		const r = cg.local.declare(cg.i32);
		const inPtr = 0;
		const luPtr = 1;
		const pivotsPtr = 2;
		const permPtr = 3;
		const m = 4;
		const n = 5;
		const batchSize = 6;
		cg.local.get(m);
		cg.local.get(n);
		cg.i32.mul();
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(matrixSize);
		cg.local.get(m);
		cg.local.get(n);
		cg.i32.lt_u();
		cg.if(cg.i32);
		cg.local.get(m);
		cg.else();
		cg.local.get(n);
		cg.end();
		cg.local.set(r);
		cg.local.get(r);
		cg.i32.const(4);
		cg.i32.mul();
		cg.local.set(pivotSize);
		cg.local.get(m);
		cg.i32.const(4);
		cg.i32.mul();
		cg.local.set(permSize);
		cg.i32.const(0);
		cg.local.set(batch);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(batch);
		cg.local.get(batchSize);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(inPtr);
		cg.local.get(batch);
		cg.local.get(matrixSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(inOffset);
		cg.local.get(luPtr);
		cg.local.get(batch);
		cg.local.get(matrixSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(luOffset);
		cg.local.get(pivotsPtr);
		cg.local.get(batch);
		cg.local.get(pivotSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(pivotsOffset);
		cg.local.get(permPtr);
		cg.local.get(batch);
		cg.local.get(permSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(permOffset);
		cg.local.get(inOffset);
		cg.local.get(luOffset);
		cg.local.get(pivotsOffset);
		cg.local.get(permOffset);
		cg.local.get(m);
		cg.local.get(n);
		cg.call(luFn);
		cg.local.get(batch);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(batch);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(batchedFn, "lu");
	return new WebAssembly.Module(cg.finish());
}

//#endregion
//#region src/backend/wasm/sort.ts
/**
* Generate a WASM function for merge sort on a single array segment.
*
* Uses bottom-up merge sort:
*   for width = 1, 2, 4, ... while width < n:
*     for i = 0, 2*width, 4*width, ... while i < n:
*       merge(arr, i, min(i+width, n), min(i+2*width, n), aux)
*
* Function signature: (arrPtr, auxPtr, n) -> void
* Both arrPtr and auxPtr point to n floats. Result ends up in arrPtr.
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_merge_sort(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	const mergeFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const k = cg.local.declare(cg.i32);
		const vi = cg.local.declare(ft);
		const vj = cg.local.declare(ft);
		const arrPtr = 0;
		const auxPtr = 1;
		const left = 2;
		const mid = 3;
		const right = 4;
		cg.local.get(left);
		cg.local.set(i);
		cg.local.get(mid);
		cg.local.set(j);
		cg.local.get(left);
		cg.local.set(k);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(k);
		cg.local.get(right);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(i);
		cg.local.get(mid);
		cg.i32.ge_u();
		cg.if(cg.void);
		cg.local.get(auxPtr);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(arrPtr);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.else();
		cg.local.get(j);
		cg.local.get(right);
		cg.i32.ge_u();
		cg.if(cg.void);
		cg.local.get(auxPtr);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(arrPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.else();
		cg.local.get(arrPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(vi);
		cg.local.get(arrPtr);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(vj);
		cg.local.get(vj);
		cg.local.get(vj);
		ft.ne();
		cg.local.get(vi);
		cg.local.get(vi);
		ft.eq();
		cg.local.get(vi);
		cg.local.get(vj);
		ft.le();
		cg.i32.and();
		cg.i32.or();
		cg.if(cg.void);
		cg.local.get(auxPtr);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(vi);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.else();
		cg.local.get(auxPtr);
		cg.local.get(k);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(vj);
		ft.store(0, 0);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.end();
		cg.end();
		cg.end();
		cg.local.get(k);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(k);
		cg.br(1);
		cg.end();
		cg.end();
	});
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const width = cg.local.declare(cg.i32);
		const i = cg.local.declare(cg.i32);
		const left = cg.local.declare(cg.i32);
		const mid = cg.local.declare(cg.i32);
		const right = cg.local.declare(cg.i32);
		const srcPtr = cg.local.declare(cg.i32);
		const dstPtr = cg.local.declare(cg.i32);
		const pass = cg.local.declare(cg.i32);
		const arrPtr = 0;
		const auxPtr = 1;
		const n = 2;
		cg.i32.const(1);
		cg.local.set(width);
		cg.i32.const(0);
		cg.local.set(pass);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(width);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.and();
		cg.i32.eqz();
		cg.if(cg.void);
		cg.local.get(arrPtr);
		cg.local.set(srcPtr);
		cg.local.get(auxPtr);
		cg.local.set(dstPtr);
		cg.else();
		cg.local.get(auxPtr);
		cg.local.set(srcPtr);
		cg.local.get(arrPtr);
		cg.local.set(dstPtr);
		cg.end();
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(i);
		cg.local.set(left);
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.add();
		cg.local.set(mid);
		cg.local.get(mid);
		cg.local.get(n);
		cg.i32.gt_u();
		cg.if(cg.void);
		cg.local.get(n);
		cg.local.set(mid);
		cg.end();
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(right);
		cg.local.get(right);
		cg.local.get(n);
		cg.i32.gt_u();
		cg.if(cg.void);
		cg.local.get(n);
		cg.local.set(right);
		cg.end();
		cg.local.get(srcPtr);
		cg.local.get(dstPtr);
		cg.local.get(left);
		cg.local.get(mid);
		cg.local.get(right);
		cg.call(mergeFn);
		cg.local.get(i);
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(width);
		cg.i32.const(2);
		cg.i32.mul();
		cg.local.set(width);
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(pass);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(pass);
		cg.i32.const(1);
		cg.i32.and();
		cg.if(cg.void);
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(arrPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(auxPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM module for batched sort.
*
* The exported function takes:
*   (inPtr, outPtr, auxPtr, n, batchSize)
*
* Where:
*   - inPtr: input array [batchSize, n]
*   - outPtr: output array [batchSize, n] (sorted)
*   - auxPtr: auxiliary buffer [n] (reused for each batch)
*   - n: length of each array to sort
*   - batchSize: number of arrays
*
* @param elementSize - Element size in bytes: 4 for f32, 8 for f64
* @returns The compiled WebAssembly.Module
*/
function createSortModule(elementSize = 4) {
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const sortFn = wasm_merge_sort(cg, ft);
	const batchedFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const batch = cg.local.declare(cg.i32);
		const arraySize = cg.local.declare(cg.i32);
		const inOffset = cg.local.declare(cg.i32);
		const outOffset = cg.local.declare(cg.i32);
		const i = cg.local.declare(cg.i32);
		const inPtr = 0;
		const outPtr = 1;
		const auxPtr = 2;
		const n = 3;
		const batchSize = 4;
		cg.local.get(n);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(arraySize);
		cg.i32.const(0);
		cg.local.set(batch);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(batch);
		cg.local.get(batchSize);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(inPtr);
		cg.local.get(batch);
		cg.local.get(arraySize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(inOffset);
		cg.local.get(outPtr);
		cg.local.get(batch);
		cg.local.get(arraySize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(outOffset);
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(outOffset);
		cg.local.get(i);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(inOffset);
		cg.local.get(i);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(outOffset);
		cg.local.get(auxPtr);
		cg.local.get(n);
		cg.call(sortFn);
		cg.local.get(batch);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(batch);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(batchedFn, "sort");
	return new WebAssembly.Module(cg.finish());
}

//#endregion
//#region src/backend/wasm/triangular-solve.ts
/**
* Generate a WASM function for upper-triangular solve (back-substitution).
*
* Solves A @ x = b where A is upper triangular.
*
* Algorithm:
*   for i = n-1 down to 0:
*     sum = b[i]
*     for j = i+1 to n-1:
*       sum -= A[i,j] * x[j]
*     x[i] = sum / A[i,i]  (or just sum if unitDiagonal)
*
* Function signature: (aPtr: i32, bPtr: i32, xPtr: i32, n: i32, unitDiagonal: i32) -> void
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_triangular_solve_upper(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const sum = cg.local.declare(ft);
		const aij = cg.local.declare(ft);
		const xj = cg.local.declare(ft);
		const aii = cg.local.declare(ft);
		const aPtr = 0;
		const bPtr = 1;
		const xPtr = 2;
		const n = 3;
		const unitDiagonal = 4;
		cg.local.get(n);
		cg.i32.const(1);
		cg.i32.sub();
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.i32.const(0);
		cg.i32.lt_s();
		cg.br_if(0);
		cg.local.get(bPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(sum);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(j);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(aPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(aij);
		cg.local.get(xPtr);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(xj);
		cg.local.get(sum);
		cg.local.get(aij);
		cg.local.get(xj);
		ft.mul();
		ft.sub();
		cg.local.set(sum);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(unitDiagonal);
		cg.i32.eqz();
		cg.if(cg.void);
		cg.local.get(aPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(i);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(aii);
		cg.local.get(sum);
		cg.local.get(aii);
		ft.div();
		cg.local.set(sum);
		cg.end();
		cg.local.get(xPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(sum);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.sub();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM function for lower-triangular solve (forward-substitution).
*
* Solves A @ x = b where A is lower triangular.
*
* Algorithm:
*   for i = 0 to n-1:
*     sum = b[i]
*     for j = 0 to i-1:
*       sum -= A[i,j] * x[j]
*     x[i] = sum / A[i,i]  (or just sum if unitDiagonal)
*
* Function signature: (aPtr: i32, bPtr: i32, xPtr: i32, n: i32, unitDiagonal: i32) -> void
*
* @param cg - The code generator
* @param ft - The float type (cg.f32 or cg.f64)
* @returns The function index
*/
function wasm_triangular_solve_lower(cg, ft) {
	const elemSize = ft.name === "f32" ? 4 : 8;
	return cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const i = cg.local.declare(cg.i32);
		const j = cg.local.declare(cg.i32);
		const sum = cg.local.declare(ft);
		const aij = cg.local.declare(ft);
		const xj = cg.local.declare(ft);
		const aii = cg.local.declare(ft);
		const aPtr = 0;
		const bPtr = 1;
		const xPtr = 2;
		const n = 3;
		const unitDiagonal = 4;
		cg.i32.const(0);
		cg.local.set(i);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(bPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(sum);
		cg.i32.const(0);
		cg.local.set(j);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(j);
		cg.local.get(i);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(aPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(j);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(aij);
		cg.local.get(xPtr);
		cg.local.get(j);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(xj);
		cg.local.get(sum);
		cg.local.get(aij);
		cg.local.get(xj);
		ft.mul();
		ft.sub();
		cg.local.set(sum);
		cg.local.get(j);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(j);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(unitDiagonal);
		cg.i32.eqz();
		cg.if(cg.void);
		cg.local.get(aPtr);
		cg.local.get(i);
		cg.local.get(n);
		cg.i32.mul();
		cg.local.get(i);
		cg.i32.add();
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		ft.load(0, 0);
		cg.local.set(aii);
		cg.local.get(sum);
		cg.local.get(aii);
		ft.div();
		cg.local.set(sum);
		cg.end();
		cg.local.get(xPtr);
		cg.local.get(i);
		cg.i32.const(elemSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.get(sum);
		ft.store(0, 0);
		cg.local.get(i);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(i);
		cg.br(1);
		cg.end();
		cg.end();
	});
}
/**
* Generate a WASM module for batched triangular solve.
*
* The exported function takes:
*   (aPtr, bPtr, xPtr, n, batchRows, numBatches, unitDiagonal, lower)
*
* Where:
*   - aPtr: [numBatches, n, n] matrix batch
*   - bPtr: [numBatches, batchRows, n] RHS vectors
*   - xPtr: [numBatches, batchRows, n] solution vectors
*   - n: matrix dimension
*   - batchRows: number of RHS vectors per matrix
*   - numBatches: number of matrices
*   - unitDiagonal: 1 if diagonal is implicitly 1
*   - lower: 1 for lower triangular, 0 for upper
*
* @param elementSize - Element size in bytes: 4 for f32, 8 for f64
* @returns The compiled WebAssembly.Module
*/
function createTriangularSolveModule(elementSize = 4) {
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const solveLower = wasm_triangular_solve_lower(cg, ft);
	const solveUpper = wasm_triangular_solve_upper(cg, ft);
	const batchedFn = cg.function([
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32,
		cg.i32
	], [], () => {
		const batch = cg.local.declare(cg.i32);
		const row = cg.local.declare(cg.i32);
		const matrixSize = cg.local.declare(cg.i32);
		const vectorSize = cg.local.declare(cg.i32);
		const aOffset = cg.local.declare(cg.i32);
		const bOffset = cg.local.declare(cg.i32);
		const xOffset = cg.local.declare(cg.i32);
		const aPtr = 0;
		const bPtr = 1;
		const xPtr = 2;
		const n = 3;
		const batchRows = 4;
		const numBatches = 5;
		const unitDiagonal = 6;
		const lower = 7;
		cg.local.get(n);
		cg.local.get(n);
		cg.i32.mul();
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(matrixSize);
		cg.local.get(n);
		cg.i32.const(elementSize);
		cg.i32.mul();
		cg.local.set(vectorSize);
		cg.i32.const(0);
		cg.local.set(batch);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(batch);
		cg.local.get(numBatches);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(aPtr);
		cg.local.get(batch);
		cg.local.get(matrixSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(aOffset);
		cg.i32.const(0);
		cg.local.set(row);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(row);
		cg.local.get(batchRows);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(bPtr);
		cg.local.get(batch);
		cg.local.get(batchRows);
		cg.i32.mul();
		cg.local.get(row);
		cg.i32.add();
		cg.local.get(vectorSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(bOffset);
		cg.local.get(xPtr);
		cg.local.get(batch);
		cg.local.get(batchRows);
		cg.i32.mul();
		cg.local.get(row);
		cg.i32.add();
		cg.local.get(vectorSize);
		cg.i32.mul();
		cg.i32.add();
		cg.local.set(xOffset);
		cg.local.get(lower);
		cg.if(cg.void);
		cg.local.get(aOffset);
		cg.local.get(bOffset);
		cg.local.get(xOffset);
		cg.local.get(n);
		cg.local.get(unitDiagonal);
		cg.call(solveLower);
		cg.else();
		cg.local.get(aOffset);
		cg.local.get(bOffset);
		cg.local.get(xOffset);
		cg.local.get(n);
		cg.local.get(unitDiagonal);
		cg.call(solveUpper);
		cg.end();
		cg.local.get(row);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(row);
		cg.br(1);
		cg.end();
		cg.end();
		cg.local.get(batch);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(batch);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(batchedFn, "triangularSolve");
	return new WebAssembly.Module(cg.finish());
}

//#endregion
//#region src/backend/wasm.ts
const moduleCache = /* @__PURE__ */ new Map();
/**
* Key for routine module cache: combines routine name and element size.
* Format: "routineName:elementSize" (e.g., "Cholesky:4" or "Cholesky:8")
*/
function routineModuleKey(name, elementSize) {
	return `${name}:${elementSize}`;
}
/** Cached WASM modules for routines (lazy-initialized), keyed by routine+dtype. */
const routineModules = /* @__PURE__ */ new Map();
/** Module creators for each WASM-accelerated routine (now accept elementSize). */
const routineModuleCreators = {
	[Routines.Cholesky]: createCholeskyModule,
	[Routines.TriangularSolve]: createTriangularSolveModule,
	[Routines.LU]: createLUModule,
	[Routines.Sort]: createSortModule,
	[Routines.Argsort]: createArgsortModule
};
/** Backend that compiles into WebAssembly bytecode for immediate execution. */
var WasmBackend = class {
	type = "wasm";
	maxArgs = 64;
	#memory;
	#nextSlot;
	#allocator;
	#buffers;
	/** Cache WebAssembly instances keyed by module for reuse in dispatch. */
	#instanceCache;
	/** Cached WASM routine instances, keyed by "routine:elementSize". */
	#routineInstances = /* @__PURE__ */ new Map();
	constructor() {
		this.#memory = new WebAssembly.Memory({ initial: 0 });
		this.#allocator = new WasmAllocator(this.#memory);
		this.#nextSlot = 1;
		this.#buffers = /* @__PURE__ */ new Map();
		this.#instanceCache = /* @__PURE__ */ new WeakMap();
	}
	malloc(size, initialData) {
		const ptr = this.#allocator.malloc(size);
		if (initialData) {
			if (initialData.byteLength !== size) throw new Error("initialData size does not match buffer size");
			new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
		}
		const slot = this.#nextSlot++;
		this.#buffers.set(slot, {
			ptr,
			size,
			ref: 1
		});
		return slot;
	}
	incRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref++;
	}
	decRef(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		buffer.ref--;
		if (buffer.ref === 0) {
			this.#allocator.free(buffer.ptr);
			this.#buffers.delete(slot);
		}
	}
	slotCount() {
		return this.#buffers.size;
	}
	async read(slot, start, count) {
		return this.readSync(slot, start, count);
	}
	readSync(slot, start, count) {
		const buffer = this.#getBuffer(slot);
		if (start === void 0) start = 0;
		if (count === void 0) count = buffer.byteLength - start;
		return buffer.slice(start, start + count);
	}
	async prepareKernel(kernel) {
		return this.prepareKernelSync(kernel);
	}
	prepareKernelSync(kernel) {
		const kernelHash = FpHash.hash(kernel);
		const module = runWithCache(moduleCache, kernelHash.toString(), () => {
			const bytes = codegenWasm(kernel);
			return new WebAssembly.Module(bytes);
		});
		return new Executable(kernel, { module });
	}
	async prepareRoutine(routine) {
		return this.prepareRoutineSync(routine);
	}
	prepareRoutineSync(routine) {
		return new Executable(routine, void 0);
	}
	dispatch(exe, inputs, outputs) {
		if (exe.source instanceof Routine) {
			const routine = exe.source;
			const dtype = routine.type.inputDtypes[0];
			const isF32 = dtype === DType.Float32;
			const isF64 = dtype === DType.Float64;
			if (isF32 || isF64) {
				const elementSize = isF32 ? 4 : 8;
				switch (routine.name) {
					case Routines.Cholesky: return this.#dispatchCholesky(routine, inputs, outputs, elementSize);
					case Routines.TriangularSolve: return this.#dispatchTriangularSolve(routine, inputs, outputs, elementSize);
					case Routines.LU: return this.#dispatchLU(routine, inputs, outputs, elementSize);
					case Routines.Sort: return this.#dispatchSort(routine, inputs, outputs, elementSize);
					case Routines.Argsort: return this.#dispatchArgsort(routine, inputs, outputs, elementSize);
				}
			}
			return runCpuRoutine(routine, inputs.map((slot) => this.#getBuffer(slot)), outputs.map((slot) => this.#getBuffer(slot)));
		}
		let instance = this.#instanceCache.get(exe.data.module);
		if (!instance) {
			instance = new WebAssembly.Instance(exe.data.module, { env: { memory: this.#memory } });
			this.#instanceCache.set(exe.data.module, instance);
		}
		const func = instance.exports.kernel;
		const ptrs = [...inputs, ...outputs].map((slot) => this.#buffers.get(slot).ptr);
		func(...ptrs);
	}
	/** Get or create a WASM instance for a routine with specific element size. */
	#getRoutineFunc(name, exportName, elementSize = 4) {
		const key = routineModuleKey(name, elementSize);
		let instance = this.#routineInstances.get(key);
		if (!instance) {
			let module = routineModules.get(key);
			if (!module) {
				const creator = routineModuleCreators[name];
				if (!creator) throw new Error(`No WASM module for routine: ${name}`);
				module = creator(elementSize);
				routineModules.set(key, module);
			}
			instance = new WebAssembly.Instance(module, { env: { memory: this.#memory } });
			this.#routineInstances.set(key, instance);
		}
		return instance.exports[exportName];
	}
	#dispatchCholesky(routine, inputs, outputs, elementSize) {
		const func = this.#getRoutineFunc(Routines.Cholesky, "cholesky", elementSize);
		const shape = routine.type.inputShapes[0];
		func(this.#buffers.get(inputs[0]).ptr, this.#buffers.get(outputs[0]).ptr, shape[shape.length - 1], shape.slice(0, -2).reduce((a, b) => a * b, 1));
	}
	#dispatchTriangularSolve(routine, inputs, outputs, elementSize) {
		const func = this.#getRoutineFunc(Routines.TriangularSolve, "triangularSolve", elementSize);
		const aShape = routine.type.inputShapes[0];
		const bShape = routine.type.inputShapes[1];
		func(this.#buffers.get(inputs[0]).ptr, this.#buffers.get(inputs[1]).ptr, this.#buffers.get(outputs[0]).ptr, aShape[aShape.length - 1], bShape[bShape.length - 2], aShape.slice(0, -2).reduce((a, b) => a * b, 1), routine.params?.unitDiagonal ? 1 : 0, routine.params?.lower ? 1 : 0);
	}
	#dispatchLU(routine, inputs, outputs, elementSize) {
		const func = this.#getRoutineFunc(Routines.LU, "lu", elementSize);
		const shape = routine.type.inputShapes[0];
		func(this.#buffers.get(inputs[0]).ptr, this.#buffers.get(outputs[0]).ptr, this.#buffers.get(outputs[1]).ptr, this.#buffers.get(outputs[2]).ptr, shape[shape.length - 2], shape[shape.length - 1], shape.slice(0, -2).reduce((a, b) => a * b, 1));
	}
	#dispatchSort(routine, inputs, outputs, elementSize) {
		const func = this.#getRoutineFunc(Routines.Sort, "sort", elementSize);
		const shape = routine.type.inputShapes[0];
		const n = shape[shape.length - 1];
		const auxPtr = this.#allocator.malloc(n * elementSize);
		func(this.#buffers.get(inputs[0]).ptr, this.#buffers.get(outputs[0]).ptr, auxPtr, n, shape.slice(0, -1).reduce((a, b) => a * b, 1));
		this.#allocator.free(auxPtr);
	}
	#dispatchArgsort(routine, inputs, outputs, elementSize) {
		const func = this.#getRoutineFunc(Routines.Argsort, "argsort", elementSize);
		const shape = routine.type.inputShapes[0];
		const n = shape[shape.length - 1];
		const auxDataPtr = this.#allocator.malloc(n * elementSize);
		const auxIdxPtr = this.#allocator.malloc(n * 4);
		func(this.#buffers.get(inputs[0]).ptr, this.#buffers.get(outputs[0]).ptr, this.#buffers.get(outputs[1]).ptr, auxDataPtr, auxIdxPtr, n, shape.slice(0, -1).reduce((a, b) => a * b, 1));
		this.#allocator.free(auxDataPtr);
		this.#allocator.free(auxIdxPtr);
	}
	/**
	* Prepare a native scan operation for efficient execution.
	* This is the unified implementation that handles all scan body types:
	* - Single kernel bodies (like cumsum)
	* - Multiple independent kernels (like Kalman filter with 2 matmuls)
	* - Bodies with data dependencies between steps
	* - Bodies where numCarry !== numY
	* - Routine steps (Cholesky, Sort) mixed with Kernel steps
	*/
	prepareNativeScanGeneral(params) {
		if (params.steps.length === 0) return null;
		try {
			const bytes = codegenNativeScanGeneral(params);
			const module = new WebAssembly.Module(bytes);
			let firstKernel = null;
			for (const step of params.steps) if (step.source instanceof Kernel) {
				firstKernel = step.source;
				break;
			}
			if (!firstKernel) firstKernel = new Kernel(0, 0, AluExp.const(DType.Float32, 0), void 0);
			const syntheticKernel = new Kernel(firstKernel.nargs, firstKernel.size, firstKernel.exp, firstKernel.reduction);
			return new Executable(syntheticKernel, { module });
		} catch (e) {
			if (DEBUG >= 1) console.warn("General native scan codegen failed:", e);
			return null;
		}
	}
	/**
	* Dispatch a general native scan operation.
	* Allocates temporary internal buffers, runs the scan, and frees them.
	*
	* @param exe - Compiled WASM executable for the scan
	* @param params - General scan parameters (includes internalSizes, auxBufferSize)
	* @param consts - Constant buffer slots
	* @param initCarry - Initial carry buffer slots
	* @param xs - Input xs buffer slots
	* @param carryOut - Output carry buffer slots
	* @param ysStacked - Output stacked ys buffer slots
	*/
	dispatchNativeScanGeneral(exe, params, consts, initCarry, xs, carryOut, ysStacked) {
		const internalSlots = [];
		for (const size of params.internalSizes) {
			const slot = this.malloc(size);
			internalSlots.push(slot);
		}
		let auxPtr = 0;
		if (params.auxBufferSize && params.auxBufferSize > 0) auxPtr = this.#allocator.malloc(params.auxBufferSize);
		try {
			let instance = this.#instanceCache.get(exe.data.module);
			if (!instance) {
				instance = new WebAssembly.Instance(exe.data.module, { env: { memory: this.#memory } });
				this.#instanceCache.set(exe.data.module, instance);
			}
			const func = instance.exports.scan;
			const ptrs = [
				...consts.map((slot) => this.#buffers.get(slot).ptr),
				...initCarry.map((slot) => this.#buffers.get(slot).ptr),
				...xs.map((slot) => this.#buffers.get(slot).ptr),
				...carryOut.map((slot) => this.#buffers.get(slot).ptr),
				...ysStacked.map((slot) => this.#buffers.get(slot).ptr),
				...internalSlots.map((slot) => this.#buffers.get(slot).ptr)
			];
			if (auxPtr) ptrs.push(auxPtr);
			func(...ptrs);
		} finally {
			for (const slot of internalSlots) this.decRef(slot);
			if (auxPtr) this.#allocator.free(auxPtr);
		}
	}
	#getBuffer(slot) {
		const buffer = this.#buffers.get(slot);
		if (!buffer) throw new SlotError(slot);
		return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
	}
};
/**
* Import all required math helper functions based on the ops used.
* Accepts either a Set<AluOp> or Map<AluOp, Set<DType>> (from distinctOps()).
* Returns a record mapping op names to WASM function indices.
*/
function importWasmHelperFuncs(cg, ops) {
	const funcs = {};
	const hasOp = (op) => ops instanceof Map ? ops.has(op) : ops.has(op);
	if (hasOp(AluOp.Sin)) funcs.sin = wasm_sin(cg);
	if (hasOp(AluOp.Cos)) funcs.cos = wasm_cos(cg);
	if (hasOp(AluOp.Asin)) funcs.asin = wasm_asin(cg);
	if (hasOp(AluOp.Atan)) funcs.atan = wasm_atan(cg);
	if (hasOp(AluOp.Exp) || hasOp(AluOp.Erf) || hasOp(AluOp.Erfc)) funcs.exp = wasm_exp(cg);
	if (hasOp(AluOp.Log)) funcs.log = wasm_log(cg);
	if (hasOp(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
	if (hasOp(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
	if (hasOp(AluOp.Threefry2x32)) funcs.threefry2x32 = wasm_threefry2x32(cg);
	return funcs;
}
/**
* Generate reduction accumulator update code.
* Assumes the expression result is on the stack.
* Leaves the accumulated value in the acc local.
*/
function codegenReductionAccumulate(cg, re, acc) {
	if (re.op === AluOp.Add) {
		cg.local.get(acc);
		if (re.dtype === DType.Bool) cg.i32.or();
		else dty(cg, re.op, re.dtype).add();
	} else if (re.op === AluOp.Mul) {
		cg.local.get(acc);
		if (re.dtype === DType.Bool) cg.i32.and();
		else dty(cg, re.op, re.dtype).mul();
	} else if (re.op === AluOp.Min || re.op === AluOp.Max) if (isFloatDtype(re.dtype)) {
		cg.local.get(acc);
		if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
		else dtyF(cg, re.op, re.dtype).max();
	} else if ([
		DType.Int32,
		DType.Uint32,
		DType.Bool
	].includes(re.dtype)) {
		const local = cg.local.declare(cg.i32);
		cg.local.tee(local);
		cg.local.get(acc);
		cg.local.get(local);
		cg.local.get(acc);
		if (re.op === AluOp.Min) if (re.dtype === DType.Int32) cg.i32.lt_s();
		else cg.i32.lt_u();
		else if (re.dtype === DType.Int32) cg.i32.gt_s();
		else cg.i32.gt_u();
		cg.select();
	} else throw new Error(`invalid reduction min/max over ${re.dtype}`);
	else throw new Error(`invalid wasm reduction op: ${re.op}`);
	cg.local.set(acc);
}
function codegenWasm(kernel) {
	const tune = tuneNullopt(kernel);
	const re = kernel.reduction;
	if (DEBUG >= 3) console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const distinctOps = mapSetUnion(tune.exp.distinctOps(), tune.epilogue?.distinctOps());
	const funcs = importWasmHelperFuncs(cg, distinctOps);
	const kernelFunc = cg.function(rep(kernel.nargs + 1, cg.i32), [], () => {
		const gidx = cg.local.declare(cg.i32);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(gidx);
		cg.i32.const(kernel.size);
		cg.i32.ge_u();
		cg.br_if(0);
		cg.local.get(kernel.nargs);
		cg.local.get(gidx);
		cg.i32.const(byteWidth(kernel.dtype));
		cg.i32.mul();
		cg.i32.add();
		if (re) {
			const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
			dty(cg, null, kernel.exp.dtype).const(re.identity);
			cg.local.set(acc);
			const ridx = cg.local.declare(cg.i32);
			cg.i32.const(0);
			cg.local.set(ridx);
			cg.loop(cg.void);
			cg.block(cg.void);
			cg.local.get(ridx);
			cg.i32.const(re.size);
			cg.i32.ge_u();
			cg.br_if(0);
			translateExp(cg, funcs, tune.exp, {
				gidx,
				ridx
			});
			codegenReductionAccumulate(cg, re, acc);
			cg.local.get(ridx);
			cg.i32.const(1);
			cg.i32.add();
			cg.local.set(ridx);
			cg.br(1);
			cg.end();
			cg.end();
			translateExp(cg, funcs, tune.epilogue, {
				acc,
				gidx
			});
		} else translateExp(cg, funcs, tune.exp, { gidx });
		dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
		cg.local.get(gidx);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(gidx);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(kernelFunc, "kernel");
	return cg.finish();
}
/**
* Generate WASM bytecode for a general native scan.
* Handles scan bodies with data dependencies and numCarry !== numY.
*
* Memory layout for function arguments:
* [...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
*  ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal)]
*/
function codegenNativeScanGeneral(params) {
	const { length, numConsts, constSizes, numCarry, carrySizes, numX, xsStrides, numY, ysStrides, internalSizes, steps, carryOutSources, yOutputSources, reverse } = params;
	const numInternal = internalSizes.length;
	if (DEBUG >= 2) console.log("codegenNativeScanGeneral params:", {
		length,
		numConsts,
		numCarry,
		numX,
		numY,
		numInternal,
		numSteps: steps.length,
		reverse
	});
	const cg = new CodeGenerator();
	cg.memory.import("env", "memory");
	const elementSize = params.elementSize ?? 4;
	const ft = elementSize === 4 ? cg.f32 : cg.f64;
	const allOps = /* @__PURE__ */ new Set();
	for (const step of steps) if (step.source instanceof Kernel) {
		const tune = tuneNullopt(step.source);
		for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
		if (tune.epilogue) for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
	}
	const funcs = importWasmHelperFuncs(cg, allOps);
	const routineFuncs = /* @__PURE__ */ new Map();
	for (const step of steps) if (step.source instanceof Routine) {
		const routineName = step.source.name;
		if (!routineFuncs.has(routineName)) {
			if (routineName === Routines.Cholesky) routineFuncs.set(routineName, wasm_cholesky(cg, ft));
			else if (routineName === Routines.Sort) routineFuncs.set(routineName, wasm_merge_sort(cg, ft));
		}
	}
	const needsAuxBuffer = params.auxBufferSize && params.auxBufferSize > 0;
	const numArgs = numConsts + numCarry + numX + numCarry + numY + numInternal + (needsAuxBuffer ? 1 : 0);
	const auxArgIdx = needsAuxBuffer ? numConsts + numCarry + numX + numCarry + numY + numInternal : -1;
	const scanFunc = cg.function(rep(numArgs, cg.i32), [], () => {
		const iter = cg.local.declare(cg.i32);
		const gidx = cg.local.declare(cg.i32);
		const copyIdx = cg.local.declare(cg.i32);
		const dataIdx = cg.local.declare(cg.i32);
		const constsBase = 0;
		const carryInBase = numConsts;
		const xsBase = numConsts + numCarry;
		const carryOutBase = numConsts + numCarry + numX;
		const ysStackedBase = numConsts + numCarry + numX + numCarry;
		const internalsBase = numConsts + numCarry + numX + numCarry + numY;
		for (let c = 0; c < numCarry; c++) {
			const size = carrySizes[c];
			cg.i32.const(0);
			cg.local.set(copyIdx);
			cg.loop(cg.void);
			cg.block(cg.void);
			cg.local.get(copyIdx);
			cg.i32.const(size);
			cg.i32.ge_u();
			cg.br_if(0);
			cg.local.get(carryOutBase + c);
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.local.get(carryInBase + c);
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.i32.load8_u(0);
			cg.i32.store8(0);
			cg.local.get(copyIdx);
			cg.i32.const(1);
			cg.i32.add();
			cg.local.set(copyIdx);
			cg.br(1);
			cg.end();
			cg.end();
		}
		cg.i32.const(0);
		cg.local.set(iter);
		cg.loop(cg.void);
		cg.block(cg.void);
		cg.local.get(iter);
		cg.i32.const(length);
		cg.i32.ge_u();
		cg.br_if(0);
		if (reverse) {
			cg.i32.const(length - 1);
			cg.local.get(iter);
			cg.i32.sub();
			cg.local.set(dataIdx);
		} else {
			cg.local.get(iter);
			cg.local.set(dataIdx);
		}
		for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
			const step = steps[stepIdx];
			const source = step.source;
			const internalIdx = step.outputInternalIdx;
			if (source instanceof Routine) {
				const routineName = source.name;
				const routineFn = routineFuncs.get(routineName);
				if (routineFn === void 0) throw new Error(`Routine ${source.name} not supported in native scan`);
				const inputSlot = step.inputSlots[0];
				if (inputSlot < numConsts) cg.local.get(constsBase + inputSlot);
				else if (inputSlot < numConsts + numCarry) cg.local.get(carryOutBase + (inputSlot - numConsts));
				else if (inputSlot < numConsts + numCarry + numX) {
					const xIdx = inputSlot - numConsts - numCarry;
					cg.local.get(xsBase + xIdx);
					cg.local.get(dataIdx);
					cg.i32.const(xsStrides[xIdx]);
					cg.i32.mul();
					cg.i32.add();
				} else {
					const prevInternalIdx = inputSlot - numConsts - numCarry - numX;
					cg.local.get(internalsBase + prevInternalIdx);
				}
				cg.local.get(internalsBase + internalIdx);
				if (routineName === Routines.Cholesky) {
					const inputShape = source.type.inputShapes[0];
					const n = inputShape[inputShape.length - 1];
					cg.i32.const(n);
					cg.call(routineFn);
				} else if (routineName === Routines.Sort) {
					const inputShape = source.type.inputShapes[0];
					const n = inputShape[inputShape.length - 1];
					const inPtrLocal = cg.local.declare(cg.i32);
					const outPtrLocal = cg.local.declare(cg.i32);
					cg.local.set(outPtrLocal);
					cg.local.set(inPtrLocal);
					cg.i32.const(0);
					cg.local.set(copyIdx);
					cg.loop(cg.void);
					cg.block(cg.void);
					cg.local.get(copyIdx);
					cg.i32.const(n);
					cg.i32.ge_u();
					cg.br_if(0);
					cg.local.get(outPtrLocal);
					cg.local.get(copyIdx);
					cg.i32.const(elementSize);
					cg.i32.mul();
					cg.i32.add();
					cg.local.get(inPtrLocal);
					cg.local.get(copyIdx);
					cg.i32.const(elementSize);
					cg.i32.mul();
					cg.i32.add();
					ft.load(0, 0);
					ft.store(0, 0);
					cg.local.get(copyIdx);
					cg.i32.const(1);
					cg.i32.add();
					cg.local.set(copyIdx);
					cg.br(1);
					cg.end();
					cg.end();
					cg.local.get(outPtrLocal);
					cg.local.get(auxArgIdx);
					cg.i32.const(n);
					cg.call(routineFn);
				}
			} else {
				const kernel = source;
				const tune = tuneNullopt(kernel);
				const re = kernel.reduction;
				cg.i32.const(0);
				cg.local.set(gidx);
				cg.loop(cg.void);
				{
					cg.block(cg.void);
					cg.local.get(gidx);
					cg.i32.const(kernel.size);
					cg.i32.ge_u();
					cg.br_if(0);
					cg.local.get(internalsBase + internalIdx);
					cg.local.get(gidx);
					cg.i32.const(byteWidth(kernel.dtype));
					cg.i32.mul();
					cg.i32.add();
					const scanCtx = {
						gidx,
						iter,
						dataIdx,
						ridx: -1,
						constsBase,
						constSizes,
						numConsts,
						xsBase,
						xsStrides,
						carryBase: carryOutBase,
						carrySizes,
						numCarry,
						internalsBase,
						internalSizes,
						numInternal,
						numInputs: numConsts + numCarry + numX
					};
					if (re) {
						const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
						dty(cg, null, kernel.exp.dtype).const(re.identity);
						cg.local.set(acc);
						const ridx = cg.local.declare(cg.i32);
						cg.i32.const(0);
						cg.local.set(ridx);
						scanCtx.ridx = ridx;
						cg.loop(cg.void);
						cg.block(cg.void);
						cg.local.get(ridx);
						cg.i32.const(re.size);
						cg.i32.ge_u();
						cg.br_if(0);
						translateExpWithGeneralScanContext(cg, funcs, tune.exp, scanCtx);
						codegenReductionAccumulate(cg, re, acc);
						cg.local.get(ridx);
						cg.i32.const(1);
						cg.i32.add();
						cg.local.set(ridx);
						cg.br(1);
						cg.end();
						cg.end();
						translateExpWithGeneralScanContext(cg, funcs, tune.epilogue, {
							...scanCtx,
							acc
						});
					} else translateExpWithGeneralScanContext(cg, funcs, tune.exp, scanCtx);
					dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
					cg.local.get(gidx);
					cg.i32.const(1);
					cg.i32.add();
					cg.local.set(gidx);
					cg.br(1);
					cg.end();
				}
				cg.end();
			}
		}
		for (let y = 0; y < numY; y++) {
			const source = yOutputSources[y];
			const yStride = ysStrides[y];
			let srcArgIdx;
			let size;
			if (source.type === "passthrough") {
				srcArgIdx = carryOutBase + source.carryIdx;
				size = carrySizes[source.carryIdx];
			} else {
				srcArgIdx = internalsBase + source.internalIdx;
				size = internalSizes[source.internalIdx];
			}
			cg.i32.const(0);
			cg.local.set(copyIdx);
			cg.loop(cg.void);
			cg.block(cg.void);
			cg.local.get(copyIdx);
			cg.i32.const(size);
			cg.i32.ge_u();
			cg.br_if(0);
			cg.local.get(ysStackedBase + y);
			cg.local.get(dataIdx);
			cg.i32.const(yStride);
			cg.i32.mul();
			cg.i32.add();
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.local.get(srcArgIdx);
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.i32.load8_u(0);
			cg.i32.store8(0);
			cg.local.get(copyIdx);
			cg.i32.const(1);
			cg.i32.add();
			cg.local.set(copyIdx);
			cg.br(1);
			cg.end();
			cg.end();
		}
		for (let c = 0; c < numCarry; c++) {
			const source = carryOutSources[c];
			const size = carrySizes[c];
			const srcLocal = source.type === "passthrough" ? carryInBase + source.carryIdx : internalsBase + source.internalIdx;
			cg.i32.const(0);
			cg.local.set(copyIdx);
			cg.loop(cg.void);
			cg.block(cg.void);
			cg.local.get(copyIdx);
			cg.i32.const(size);
			cg.i32.ge_u();
			cg.br_if(0);
			cg.local.get(carryOutBase + c);
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.local.get(srcLocal);
			cg.local.get(copyIdx);
			cg.i32.add();
			cg.i32.load8_u(0);
			cg.i32.store8(0);
			cg.local.get(copyIdx);
			cg.i32.const(1);
			cg.i32.add();
			cg.local.set(copyIdx);
			cg.br(1);
			cg.end();
			cg.end();
		}
		cg.local.get(iter);
		cg.i32.const(1);
		cg.i32.add();
		cg.local.set(iter);
		cg.br(1);
		cg.end();
		cg.end();
	});
	cg.export(scanFunc, "scan");
	return cg.finish();
}
/**
* Translate an AluExp to WASM code within a general scan context.
* Supports reading from internal buffers (for data dependencies between steps).
*/
function translateExpWithGeneralScanContext(cg, funcs, exp, ctx) {
	const references = /* @__PURE__ */ new Map();
	const seen = /* @__PURE__ */ new Set();
	const countReferences = (e) => {
		references.set(e, (references.get(e) ?? 0) + 1);
		if (!seen.has(e)) {
			seen.add(e);
			for (const src of e.src) countReferences(src);
		}
	};
	const expContext = /* @__PURE__ */ new Map();
	const gen = (e) => {
		if (expContext.has(e)) {
			cg.local.get(expContext.get(e));
			return;
		}
		const { op, src, dtype, arg } = e;
		if (op === AluOp.GlobalIndex) {
			const gid = arg[0];
			const bw = byteWidth(dtype);
			if (gid < ctx.numConsts) cg.local.get(ctx.constsBase + gid);
			else if (gid < ctx.numConsts + ctx.numCarry) {
				const carryIdx = gid - ctx.numConsts;
				cg.local.get(ctx.carryBase + carryIdx);
			} else if (gid < ctx.numInputs) {
				const xIdx = gid - ctx.numConsts - ctx.numCarry;
				cg.local.get(ctx.xsBase + xIdx);
				cg.local.get(ctx.dataIdx);
				cg.i32.const(ctx.xsStrides[xIdx]);
				cg.i32.mul();
				cg.i32.add();
			} else {
				const internalIdx = gid - ctx.numInputs;
				cg.local.get(ctx.internalsBase + internalIdx);
			}
			gen(src[0]);
			cg.i32.const(bw);
			cg.i32.mul();
			cg.i32.add();
			dty(cg, op, dtype).load(Math.log2(bw));
		} else if (op === AluOp.Add) {
			gen(src[0]);
			gen(src[1]);
			if (dtype === DType.Bool) cg.i32.or();
			else dty(cg, op, dtype).add();
		} else if (op === AluOp.Sub) {
			gen(src[0]);
			gen(src[1]);
			dty(cg, op, dtype).sub();
		} else if (op === AluOp.Mul) {
			gen(src[0]);
			gen(src[1]);
			if (dtype === DType.Bool) cg.i32.and();
			else dty(cg, op, dtype).mul();
		} else if (op === AluOp.Idiv) {
			gen(src[0]);
			gen(src[1]);
			if (dtype === DType.Int32) cg.i32.div_s();
			else cg.i32.div_u();
		} else if (op === AluOp.Mod) {
			gen(src[0]);
			gen(src[1]);
			if (dtype === DType.Int32) cg.i32.rem_s();
			else cg.i32.rem_u();
		} else if (op === AluOp.Min) {
			gen(src[0]);
			gen(src[1]);
			if (isFloatDtype(dtype)) dtyF(cg, op, dtype).min();
			else throw new Error("integer min in scan not yet supported");
		} else if (op === AluOp.Max) {
			gen(src[0]);
			gen(src[1]);
			if (isFloatDtype(dtype)) dtyF(cg, op, dtype).max();
			else throw new Error("integer max in scan not yet supported");
		} else if (op === AluOp.Const) dty(cg, op, dtype).const(arg);
		else if (op === AluOp.Special && arg[0] === "gidx") cg.local.get(ctx.gidx);
		else if (op === AluOp.Special && arg[0] === "ridx") {
			if (ctx.ridx < 0) throw new Error("ridx used but not in reduction context");
			cg.local.get(ctx.ridx);
		} else if (op === AluOp.Variable && arg === "gidx") cg.local.get(ctx.gidx);
		else if (op === AluOp.Variable && arg === "ridx") {
			if (ctx.ridx < 0) throw new Error("ridx used but not in reduction context");
			cg.local.get(ctx.ridx);
		} else if (op === AluOp.Variable && arg === "acc") {
			if (ctx.acc === void 0) throw new Error("acc used but not in epilogue context");
			cg.local.get(ctx.acc);
		} else if (op === AluOp.Reciprocal) {
			dtyF(cg, op, dtype).const(1);
			gen(src[0]);
			dtyF(cg, op, dtype).div();
		} else if (op === AluOp.Sqrt) {
			gen(src[0]);
			dtyF(cg, op, dtype).sqrt();
		} else if (op === AluOp.Cmplt) {
			gen(src[0]);
			gen(src[1]);
			const srcDtype = src[0].dtype;
			if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
			else if (srcDtype === DType.Int32) cg.i32.lt_s();
			else cg.i32.lt_u();
		} else if (op === AluOp.Cmpne) {
			gen(src[0]);
			gen(src[1]);
			dty(cg, op, src[0].dtype).ne();
		} else if (op === AluOp.Where) {
			gen(src[1]);
			gen(src[2]);
			gen(src[0]);
			cg.select();
		} else if (op === AluOp.Cast) {
			gen(src[0]);
			const srcDtype = src[0].dtype;
			if (srcDtype === dtype) {} else if (srcDtype === DType.Float32 && dtype === DType.Int32) cg.i32.trunc_sat_f32_s();
			else if (srcDtype === DType.Int32 && dtype === DType.Float32) cg.f32.convert_i32_s();
			else if (srcDtype === DType.Float32 && dtype === DType.Float64) cg.f64.promote_f32();
			else if (srcDtype === DType.Float64 && dtype === DType.Float32) cg.f32.demote_f64();
			else if (srcDtype === DType.Uint32 && dtype === DType.Float32) cg.f32.convert_i32_u();
			else if (srcDtype === DType.Float32 && dtype === DType.Uint32) cg.i32.trunc_sat_f32_u();
			else throw new Error(`Cast ${srcDtype} -> ${dtype} in scan not yet supported`);
		} else throw new Error(`translateExpWithGeneralScanContext: unsupported op ${op}`);
		if ((references.get(e) ?? 0) > 1) {
			const local = cg.local.declare(dty(cg, op, dtype));
			cg.local.tee(local);
			expContext.set(e, local);
		}
	};
	countReferences(exp);
	gen(exp);
}
function translateExp(cg, funcs, exp, ctx) {
	const references = /* @__PURE__ */ new Map();
	const seen = /* @__PURE__ */ new Set();
	const countReferences = (exp$1) => {
		references.set(exp$1, (references.get(exp$1) ?? 0) + 1);
		if (!seen.has(exp$1)) {
			seen.add(exp$1);
			for (const src of exp$1.src) countReferences(src);
		}
	};
	const expContext = /* @__PURE__ */ new Map();
	const gen = (exp$1) => {
		if (expContext.has(exp$1)) return cg.local.get(expContext.get(exp$1));
		const { op, src, dtype, arg } = exp$1;
		if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
			gen(src[0]);
			gen(src[1]);
			if (op === AluOp.Add) if (dtype === DType.Bool) cg.i32.or();
			else dty(cg, op, dtype).add();
			else if (op === AluOp.Sub) dty(cg, op, dtype).sub();
			else if (op === AluOp.Mul) if (dtype === DType.Bool) cg.i32.and();
			else dty(cg, op, dtype).mul();
			else if (op === AluOp.Idiv) if (isFloatDtype(dtype)) {
				dtyF(cg, op, dtype).div();
				dtyF(cg, op, dtype).trunc();
			} else if (dtype === DType.Uint32) cg.i32.div_u();
			else if (dtype === DType.Int32) cg.i32.div_s();
			else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === AluOp.Mod) if (isFloatDtype(dtype)) {
				const dt = dtyF(cg, op, dtype);
				const a = cg.local.declare(dt);
				const b = cg.local.declare(dt);
				cg.local.set(b);
				cg.local.tee(a);
				cg.local.get(a);
				cg.local.get(b);
				dt.div();
				dt.trunc();
				cg.local.get(b);
				dt.mul();
				dt.sub();
			} else if (dtype === DType.Uint32) cg.i32.rem_u();
			else if (dtype === DType.Int32) cg.i32.rem_s();
			else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === AluOp.Min || op === AluOp.Max) if (isFloatDtype(dtype)) if (op === AluOp.Min) dtyF(cg, op, dtype).min();
			else dtyF(cg, op, dtype).max();
			else if (dtype === DType.Int32 || dtype === DType.Uint32 || dtype === DType.Bool) {
				const a = cg.local.declare(cg.i32);
				const b = cg.local.declare(cg.i32);
				cg.local.set(b);
				cg.local.tee(a);
				cg.local.get(b);
				cg.local.get(a);
				cg.local.get(b);
				if (dtype === DType.Int32) if (op === AluOp.Min) cg.i32.lt_s();
				else cg.i32.gt_s();
				else if (op === AluOp.Min) cg.i32.lt_u();
				else cg.i32.gt_u();
				cg.select();
			} else throw new UnsupportedOpError(op, dtype, "wasm");
			else if (op === AluOp.Cmplt) {
				const srcDtype = src[0].dtype;
				if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
				else if (srcDtype === DType.Int32) cg.i32.lt_s();
				else if (srcDtype === DType.Uint32) cg.i32.lt_u();
				else throw new UnsupportedOpError(op, dtype, "wasm");
			} else if (op === AluOp.Cmpne) dty(cg, op, src[0].dtype).ne();
			else throw new UnsupportedOpError(op, dtype, "wasm");
		} else if (AluGroup.Unary.has(op)) {
			const callFuncF32 = (func) => {
				if (dtype !== DType.Float32) if (dtype === DType.Float64) cg.f32.demote_f64();
				else throw new UnsupportedOpError(op, dtype, "wasm");
				cg.call(func);
				if (dtype === DType.Float64) cg.f64.promote_f32();
			};
			if (op === AluOp.Sin) gen(src[0]), callFuncF32(funcs.sin);
			else if (op === AluOp.Cos) gen(src[0]), callFuncF32(funcs.cos);
			else if (op === AluOp.Asin) gen(src[0]), callFuncF32(funcs.asin);
			else if (op === AluOp.Atan) gen(src[0]), callFuncF32(funcs.atan);
			else if (op === AluOp.Exp) gen(src[0]), callFuncF32(funcs.exp);
			else if (op === AluOp.Log) gen(src[0]), callFuncF32(funcs.log);
			else if (op === AluOp.Erf) gen(src[0]), callFuncF32(funcs.erf);
			else if (op === AluOp.Erfc) gen(src[0]), callFuncF32(funcs.erfc);
			else if (op === AluOp.Sqrt) gen(src[0]), dtyF(cg, op, dtype).sqrt();
			else if (op === AluOp.Reciprocal) {
				const dt = dtyF(cg, op, dtype);
				dt.const(1), gen(src[0]), dt.div();
			} else if (op === AluOp.Floor) gen(src[0]), dtyF(cg, op, dtype).floor();
			else if (op === AluOp.Ceil) gen(src[0]), dtyF(cg, op, dtype).ceil();
			else if (op === AluOp.Cast) {
				gen(src[0]);
				const dtype0 = src[0].dtype;
				const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32 || dtype0 === DType.Bool;
				if (dtype === DType.Int32) if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_s();
				else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_s();
				else if (i32repr);
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === DType.Uint32) if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_u();
				else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_u();
				else if (i32repr);
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === DType.Float32) if (dtype0 === DType.Float32);
				else if (dtype0 === DType.Float64) cg.f32.demote_f64();
				else if (dtype0 === DType.Int32 || dtype0 === DType.Bool) cg.f32.convert_i32_s();
				else if (dtype0 === DType.Uint32) cg.f32.convert_i32_u();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === DType.Float64) if (dtype0 === DType.Float32) cg.f64.promote_f32();
				else if (dtype0 === DType.Float64);
				else if (dtype0 === DType.Int32 || dtype0 === DType.Bool) cg.f64.convert_i32_s();
				else if (dtype0 === DType.Uint32) cg.f64.convert_i32_u();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else if (dtype === DType.Bool) if (dtype0 === DType.Bool);
				else if (i32repr) cg.i32.const(0), cg.i32.ne();
				else if (dtype0 === DType.Float32) cg.f32.const(0), cg.f32.ne();
				else if (dtype0 === DType.Float64) cg.f64.const(0), cg.f64.ne();
				else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
				else throw new UnsupportedOpError(op, dtype, "wasm");
			} else if (op === AluOp.Bitcast) {
				gen(src[0]);
				const dtype0 = src[0].dtype;
				if (dtype !== dtype0) {
					const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32;
					if (dtype === DType.Int32 || dtype === DType.Uint32) if (dtype0 === DType.Float32) cg.i32.reinterpret_f32();
					else if (i32repr);
					else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
					else if (dtype === DType.Float32) if (i32repr) cg.f32.reinterpret_i32();
					else if (dtype0 === DType.Float32);
					else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
					else throw new UnsupportedOpError(op, dtype, "wasm");
				}
			} else throw new UnsupportedOpError(op, dtype, "wasm");
		} else if (op === AluOp.Where) {
			gen(src[1]);
			gen(src[2]);
			gen(src[0]);
			cg.select();
		} else if (op === AluOp.Threefry2x32) {
			for (let i = 0; i < 4; i++) gen(src[i]);
			cg.call(funcs.threefry2x32);
			if (arg === "xor") cg.i32.xor();
			else if (arg === 0) cg.drop();
			else if (arg === 1) {
				const local = cg.local.declare(cg.i32);
				cg.local.set(local);
				cg.drop();
				cg.local.get(local);
			} else throw new UnsupportedOpError(op, dtype, "wasm", arg);
		} else if (op === AluOp.Const) return dty(cg, op, dtype).const(arg);
		else if (op === AluOp.Special) return cg.local.get(ctx[arg[0]]);
		else if (op === AluOp.Variable) return cg.local.get(ctx[arg]);
		else if (op === AluOp.GlobalIndex) {
			const [gid, len] = arg;
			gen(src[0]);
			const local = cg.local.declare(cg.i32);
			cg.local.tee(local);
			cg.i32.const(0);
			cg.local.get(local), cg.i32.const(len), cg.i32.lt_u();
			cg.select();
			cg.i32.const(byteWidth(dtype));
			cg.i32.mul();
			cg.local.get(gid);
			cg.i32.add();
			dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
		} else throw new UnsupportedOpError(op, dtype, "wasm");
		if ((references.get(exp$1) ?? 0) > 1) {
			const local = cg.local.declare(dty(cg, op, dtype));
			cg.local.tee(local);
			expContext.set(exp$1, local);
		}
	};
	countReferences(exp);
	gen(exp);
}
function dty(cg, op, dtype) {
	switch (dtype) {
		case DType.Float32: return cg.f32;
		case DType.Float64: return cg.f64;
		case DType.Int32:
		case DType.Uint32:
		case DType.Bool: return cg.i32;
		default: throw new UnsupportedOpError(op, dtype, "wasm");
	}
}
function dtyF(cg, op, dtype) {
	switch (dtype) {
		case DType.Float32: return cg.f32;
		case DType.Float64: return cg.f64;
		default: throw new UnsupportedOpError(op, dtype, "wasm");
	}
}

//#endregion
//#region src/backend.ts
const devices = [
	"cpu",
	"wasm",
	"webgpu",
	"webgl"
];
const initializedBackends = /* @__PURE__ */ new Map();
initializedBackends.set("cpu", new CpuBackend());
if (typeof WebAssembly !== "undefined") initializedBackends.set("wasm", new WasmBackend());
let defaultBackend = initializedBackends.has("wasm") ? "wasm" : "cpu";
/** Configure the default device for arrays. */
function defaultDevice(device) {
	if (device !== void 0) if (initializedBackends.has(device)) defaultBackend = device;
	else throw new Error(`Backend not initialized: ${device}`);
	return defaultBackend;
}
/**
* Initialize `jax-js` library backends.
*
* By default, this will initialize all available backends. If one or more
* backends is provided, only attempt to initialize those. Returns a list of
* available backends.
*/
async function init(...devicesToInit) {
	if (devicesToInit.length === 0) devicesToInit = devices;
	const promises = [];
	for (const device of new Set(devicesToInit)) if (!initializedBackends.has(device)) promises.push((async () => {
		const backend = await createBackend(device);
		if (backend) initializedBackends.set(device, backend);
	})());
	await Promise.all(promises);
	return Array.from(initializedBackends.keys());
}
/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(device) {
	if (device === "cpu") return new CpuBackend();
	else if (device === "wasm") {
		if (typeof WebAssembly === "undefined") return null;
		return new WasmBackend();
	} else if (device === "webgpu") {
		if (!navigator.gpu) return null;
		const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!adapter) return null;
		const { WebGPUBackend } = await import("./webgpu-Djs3FJCL.js");
		const importantLimits = [
			"maxBufferSize",
			"maxComputeInvocationsPerWorkgroup",
			"maxComputeWorkgroupSizeX",
			"maxComputeWorkgroupSizeY",
			"maxComputeWorkgroupSizeZ",
			"maxComputeWorkgroupStorageSize",
			"maxComputeWorkgroupsPerDimension",
			"maxStorageBufferBindingSize",
			"maxStorageBuffersPerShaderStage",
			"maxStorageTexturesPerShaderStage"
		];
		const requestedFeatures = ["shader-f16", "timestamp-query"];
		try {
			const device$1 = await adapter.requestDevice({
				requiredLimits: Object.fromEntries(importantLimits.map((limit) => [limit, adapter.limits[limit]])),
				requiredFeatures: requestedFeatures.filter((feature) => adapter.features.has(feature))
			});
			return new WebGPUBackend(device$1);
		} catch (error) {
			console.error("Unexpected error requesting WebGPU device:", error);
			return null;
		}
	} else if (device === "webgl") {
		if (typeof WebGL2RenderingContext === "undefined") return null;
		const canvas = new OffscreenCanvas(0, 0);
		const gl = canvas.getContext("webgl2", {
			alpha: false,
			antialias: false,
			premultipliedAlpha: false,
			preserveDrawingBuffer: false,
			depth: false,
			stencil: false,
			failIfMajorPerformanceCaveat: true
		});
		if (!gl) return null;
		if (!gl.getExtension("EXT_color_buffer_float")) return null;
		const { WebGLBackend } = await import("./webgl-ZEd4_JXf.js");
		return new WebGLBackend(gl);
	} else throw new Error(`Backend not found: ${device}`);
}
/** Retrieve a backend that has been initialized. */
function getBackend(device) {
	device = device ?? defaultBackend;
	const backend = initializedBackends.get(device);
	if (!backend) throw new Error(`${device} backend not ready, call init() first`);
	return backend;
}
var Executable = class {
	constructor(source, data) {
		this.source = source;
		this.data = data;
	}
};
var SlotError = class extends Error {
	constructor(slot) {
		super(`Used a buffer that is invalid or already freed: ${slot}`);
	}
};
var UnsupportedOpError = class extends Error {
	constructor(op, dtype, device, arg) {
		let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
		if (arg !== void 0) msg += ` with arg ${JSON.stringify(arg)}`;
		super(msg);
	}
};
var UnsupportedRoutineError = class extends Error {
	constructor(name, device) {
		super(`routine '${name}' is not supported in ${device} backend`);
	}
};

//#endregion
export { AluExp, AluGroup, AluOp, AluVar, DEBUG, DType, Executable, FpHash, Kernel, PPrint, Reduction, Routine, Routines, ShapeTracker, SlotError, UnsupportedOpError, UnsupportedRoutineError, accessorAluExp, accessorGlobal, assertNonNull, byteWidth, checkAxis, checkInts, deepEqual, defaultDevice, devices, dtypedArray, dtypedJsArray, findPow2, generalBroadcast, getBackend, init, invertPermutation, isFloatDtype, isNumberPair, isPermutation, mapSetUnion, normalizeAxis, partitionList, prod, promoteTypes, range, recursiveFlatten, rep, reportScanPath, runWithCache, setDebug, setScanPathCallback, strip1, toposort, tuneNullopt, tuneWebgpu, unravelAlu, unzip2, zip, zipn };