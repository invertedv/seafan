package seafan

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	flt "gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

const (
	// delimiter for strings below
	delim = "$"

	// ifs is a list of relations for If statements
	ifs = ">$>=$<$<=$==$!="

	// functions is a list of implemented functions
	functions = "log$exp$lag$pow$if$sum$mean$max$min$s$median$count$cuma$counta$cumb$countb"

	// funArgs is a list of the number of arguments that functions take
	funArgs = "1$1$2$2$3$1$1$1$1$1$1$1$2$1$2$1"

	// funLevels indicates whether the function is calculated at the row level or is a summary.
	funLevels = "R$R$R$R$R$S$S$S$S$S$S$S$R$R$R$R"

	// logicals are disjunctions, conjunctions
	logicals = "&&$||"

	// Comparisons are comparison operators
	comparisons = ">$>=$<$<=$==$!="

	arith1 = "+$-"

	arith2 = "*$/"

	arith3 = "^"

	// arithmetics is a list operators
	arithmetics = arith1 + "$" + arith2 + "$" + arith3

	// operations is a list of supported operations
	operations = logicals + "$" + comparisons + "$" + arithmetics
)

// OpNode is a single node of an expression.
// The input expression is successively broken into simpler expressions.  Each leaf is devoid of expressions--they
// are only values.  Hence, leaves have no Inputs.
//
// operations:
// If the expression at a node is an operation, it will be broken into two subexpressions. The subexpressions are
// determined by scanning from the left using the order of precedence (+,-,*,/), respecting parentheses. The two
// subexpressions create two new nodes in Inputs.
//
// Functions:
// If the expression is a function, each argument is assigned to an Input (in order).  Functions have at least one
// input (argument). Two types of functions are supported: those that operate at the row level and those that
// operate at the summary level.  A row-level function will create a slice that has as many elements as the Pipeline.
// A summary-level function, such as "mean", will have a single element.
//
// Available row-level functions are:
//   - exp(<expr>)
//   - log(<expr>)
//   - lag(<expr>,<missing>), where <missing> is used for the first element.
//   - if(<test>, <true>, <false>), where the value <yes> is used if <condition> is greater than 0 and <false> o.w.
//   - counta(<expr>), countb(<expr>) is the number of rows after (before) the current row.
//   - cuma(<expr>,<missing>), cumrb(<expr>,<missing>) is the cumulative sum of <expr> after (before) the current row
//     and <missing> is used for the last (first) element.
//
// The values in <...> can be any expression.
//
// Available summary-level functions are:
//   - mean(<expr>)
//   - median(<expr>)
//   - count(<expr>)
//   - sum(<expr>)
//   - max(<expr>)
//   - min(<expr>)
//
// Logical operators are supported:
//   - &&  and
//   - ||  or
//
// Logical operators resolve to 0 or 1.
type OpNode struct {
	Expression string    // expression this node implements
	Value      []float64 // node value. Value is nil until Evaluate is run
	Neg        bool      // negate result when populating Value
	FunName    string    // name of function/operation to apply to the Inputs. The value is "" for leaves.
	Inputs     []*OpNode // Inputs to node calculation
}

// negLocation determines where to place a leading minus sign.
//   - 0  there is no leading minus sign
//   - 1  on the current Node
//   - 2  on Inputs[0]
func negLocation(expr string) int {
	if expr == "" {
		return 0
	}

	if expr[0] != '-' {
		return 0
	}

	if _, allIn := allInParen(expr); allIn {
		return 1
	}

	if _, args := searchOp(expr, arith1); args != nil {
		return 2
	}

	if _, args := searchOp(expr, arith2); args != nil {
		return 1
	}

	if _, args := searchOp(expr, arith3); args != nil {
		return 1
	}

	return 1
}

// Expr2Tree builds the OpNode tree that is a binary representation of an expression.
// The process to add a field to a Pipeline is:
//  1. Create the *OpNode tree to evaluate the expression using Expr2Tree
//  2. Populate the values from a Pipeline using Evaluate.
//  3. Add the values to the Pipeline using AddToPipe
//
// Note, you can access the values after Evaluate without adding the field to the Pipeline from the Value element
// of the root node.
//
// The expression can include:
//   - arithmetic operators: +, -, *, /
//   - exponentation: ^
//   - functions: log, exp
//   - logicals: &&, ||.  These evaluate to 0 or 1.
//   - if statements: if(condition, value if true, value if false). The true value is applied if the condition evaluates
//     to a positive value.
//   - parentheses
func Expr2Tree(curNode *OpNode) error {
	curNode.Expression = strings.ReplaceAll(curNode.Expression, " ", "")

	if e := matchedParen(curNode.Expression); e != nil {
		return e
	}

	curNode.Expression, _ = allInParen(curNode.Expression)

	// check for a leading minus sign and where to place it on our tree
	negLoc := negLocation(curNode.Expression)

	if negLoc > 0 {
		if curNode.Expression[0] == '-' {
			curNode.Expression = curNode.Expression[1:]
		}

		if negLoc == 1 {
			curNode.Neg = true
		}

		// need to check again
		curNode.Expression, _ = allInParen(curNode.Expression)
	}

	// divide into an operation or function & arguments
	op, args, err := splitExpr(curNode.Expression)
	if err != nil {
		return err
	}

	// nothing to do (leaf)
	if op == "" {
		return nil
	}

	// if an op is a negative, recast it as added the negative of the first term
	if op == "-" {
		op = "+"
		args[1] = "-" + args[1]
	}

	curNode.FunName = op
	if args == nil {
		return nil
	}

	curNode.Inputs = make([]*OpNode, len(args))
	for ind := 0; ind < len(curNode.Inputs); ind++ {
		curNode.Inputs[ind] = &OpNode{Neg: false}

		curNode.Inputs[ind].Expression = args[ind]
		if e := Expr2Tree(curNode.Inputs[ind]); e != nil {
			return e
		}
	}

	// if there is only 1 Input, set curNode.Neg, o.w. set first term, Inputs[0].Neg
	if negLoc == 2 {
		curNode.Inputs[0].Neg = true
	}

	return nil
}

// find the first needle that is not within parens.  Ignore the first character--that cannot be a true operator.
// Needles string uses delim to separate the needles
func searchOp(expr, needles string) (op string, args []string) {
	if expr == "" {
		return "", []string{expr}
	}

	ignore := 0
	for indx := 0; indx < len(expr)-1; indx++ {
		// needles can be 1 or 2 characters wide
		ch := expr[indx : indx+1]
		ch2 := expr[indx : indx+2]
		switch ch {
		case "(":
			ignore++
		case ")":
			ignore--
		default:
			if ignore == 0 && indx > 0 {
				// check 2-character needles first
				if checkSlice(ch2, needles) {
					return ch2, []string{expr[0:indx], expr[indx+2:]}
				}

				if checkSlice(ch, needles) {
					return ch, []string{expr[0:indx], expr[indx+1:]}
				}
			}
		}
	}

	return "", nil
}

// getArgs breaks up function arguments into elements of a slice
func getArgs(inner string) (pieces []string) {
	pieces = nil
	ok := true

	for ok {
		_, arg := searchOp(inner, ",")

		if arg == nil {
			ok = false
			pieces = append(pieces, inner)
			continue
		}

		pieces = append(pieces, arg[0])
		inner = arg[1]
	}

	return pieces
}

// functionDetails returns # of arguments and level (row, summary) of the function.
// It returns 0, "" if the function isn't found
func functionDetails(funName string) (argNo int, funLevel string) {
	args := strings.Split(funArgs, delim)
	levels := strings.Split(funLevels, delim)

	for ind, f := range strings.Split(functions, delim) {
		if f != funName {
			continue
		}
		expInt64, _ := strconv.ParseInt(args[ind], 10, 64)
		expInt := int(expInt64)

		level := "row"
		if levels[ind] == "S" {
			level = "summary"
		}

		return expInt, level
	}

	return 0, ""
}

// getFunction determines if expr is a function call.
//   - If it is, it returns the function and arguments.
//   - If it is not, it returns funName=""
func getFunction(expr string) (funName string, args []string, err error) {
	// need a paren for a function call
	if expr == "" || !strings.Contains(expr, "(") {
		return "", nil, nil
	}

	indx := strings.Index(expr, "(")

	// If not a single function call, more parsing must be done first.
	inner, allIn := allInParen(expr[indx:])
	if !allIn {
		return "", nil, nil
	}

	// a function will have only alphas before the left paren
	f := strings.ToLower(expr[0:indx])
	for ind := 0; ind < indx; ind++ {
		if !strings.Contains("abcdefghijklmnopqrstuvwxyz", f[ind:ind+1]) {
			return "", nil, nil
		}
	}

	// Is this a known function?
	if !checkSlice(f, functions) {
		return f, nil, fmt.Errorf("unknown function: %s", f)
	}

	// get arguments
	args = getArgs(inner)
	if numArg, _ := functionDetails(f); numArg != len(args) && numArg > 0 {
		return f, args, fmt.Errorf("wrong number of arguments in %s", f)
	}

	return f, args, nil
}

// allInParen checks if entire expr is within parens.  If it is, the unneeded parens are stripped off.
func allInParen(expr string) (inner string, allIn bool) {
	if expr == "" {
		return expr, false
	}

	if expr[0] != '(' {
		return expr, false
	}

	// find matching paren
	depth := 1
	for ind := 1; ind < len(expr); ind++ {
		if expr[ind] == '(' {
			depth++
		}
		if expr[ind] == ')' {
			depth--
		}
		if depth == 0 {
			if ind+1 == len(expr) {
				// there may be more...
				inner, _ = allInParen(expr[1:ind])
				return inner, true
			}
			return expr, false
		}
	}

	return expr, false
}

// splitExpr does the following:
//   - if expr is an operation, it splits this into two sub-operations
//   - if expr is a function call, it splits it into its arguments
//
// The operations/arguments are loaded into the Inputs array.
func splitExpr(expr string) (op string, args []string, err error) {
	if expr == "" {
		return "", nil, nil
	}

	// If this is a function, we will create a node just to calculate it and then recurse to get the argument
	if op, args, err = getFunction(expr); op != "" {
		return op, args, err
	}

	// order of precedence: logicals -> comparisons -> +- -> */ -> ^

	op, args = searchOp(expr, logicals)
	if args != nil {
		return op, args, nil
	}

	op, args = searchOp(expr, comparisons)
	if args != nil {
		return op, args, nil
	}

	// break into two parts at the first +/- not nested in parens, ignoring the first character
	op, args = searchOp(expr, arith1)
	if args != nil {
		return op, args, nil
	}

	op, args = searchOp(expr, arith2)
	if args != nil {
		return op, args, nil
	}

	op, args = searchOp(expr, "^")
	if args != nil {
		return op, args, nil
	}

	return "", []string{expr}, nil
}

// ifCond evaluates an "if" condition.
func ifCond(node *OpNode) error {
	var deltas []int
	node.Value, deltas = getDeltas(node.Inputs)

	// do some checking on lengths
	indT, indF := 0, 0
	for indRes := 0; indRes < len(node.Value); indRes++ {
		x := node.Inputs[2].Value[indF]
		if node.Inputs[0].Value[indRes] > 0.0 {
			x = node.Inputs[1].Value[indT]
		}
		node.Value[indRes] = x
		indT += deltas[1]
		indF += deltas[2]
	}

	return nil
}

// checkSlice returns true of needle is in haystack
func checkSlice(needle, haystack string) bool {
	for _, straw := range strings.Split(haystack, delim) {
		if needle == straw {
			return true
		}
	}

	return false
}

// getDeltas returns an array for the results and a slice of increments for moving through the Inputs
func getDeltas(inputs []*OpNode) (x []float64, deltas []int) {
	if inputs == nil {
		return nil, nil
	}

	n := 1
	for ind := 0; ind < len(inputs); ind++ {
		if inputs[ind].Value == nil {
			return nil, nil
		}

		d := 0
		nx := len(inputs[ind].Value)
		if nx > 1 {
			d = 1
			if nx > n {
				n = nx
			}
		}

		deltas = append(deltas, d)
	}

	return make([]float64, n), deltas
}

// EvalSFunction evaluates a summary function.
func EvalSFunction(node *OpNode) error {
	const medianQ = 0.5

	switch node.FunName {
	case "sum":
		node.Value = []float64{flt.Sum(node.Inputs[0].Value)}
	case "max":
		node.Value = []float64{flt.Max(node.Inputs[0].Value)}
	case "min":
		node.Value = []float64{flt.Min(node.Inputs[0].Value)}
	case "mean":
		node.Value = []float64{stat.Mean(node.Inputs[0].Value, nil)}
	case "s":
		node.Value = []float64{stat.StdDev(node.Inputs[0].Value, nil)}
	case "median":
		node.Value = []float64{stat.Quantile(medianQ, stat.Empirical, node.Inputs[0].Value, nil)}
	case "count":
		node.Value = []float64{float64(len(node.Inputs[0].Value))}
	default:
		return fmt.Errorf("unknown function: %s", node.FunName)
	}

	goNegative(node.Value, node.Neg)

	return nil
}

// evalFunction evaluates a function call
func evalFunction(node *OpNode) error {
	if !checkSlice(node.FunName, functions) {
		return fmt.Errorf("%s function not implemented", node.FunName)
	}

	if node.FunName == "if" {
		if e := ifCond(node); e != nil {
			return e
		}
		goNegative(node.Value, node.Neg)

		return nil
	}

	var deltas []int

	node.Value, deltas = getDeltas(node.Inputs)

	ind1 := len(node.Inputs[0].Value) - 1
	two := len(deltas) > 1
	var ind2 int
	if two {
		if len(node.Inputs) > 1 {
			ind2 = len(node.Inputs[1].Value) - 1
		}
	}

	if _, funLevel := functionDetails(node.FunName); funLevel == "summary" {
		if e := EvalSFunction(node); e != nil {
			return e
		}
	}

	// These will be Row functions
	// move backwards for the lag function
	for ind := len(node.Value) - 1; ind >= 0; ind-- {
		switch node.FunName {
		case "cuma":
			if ind < len(node.Value)-1 {
				node.Value[ind] = flt.Sum(node.Inputs[0].Value[ind+1:])
			} else {
				node.Value[ind] = node.Inputs[1].Value[0]
			}
		case "counta":
			node.Value[ind] = float64(len(node.Value) - ind - 1)
		case "cumb":
			if ind > 0 {
				node.Value[ind] = flt.Sum(node.Inputs[0].Value[:ind])
			} else {
				node.Value[ind] = node.Inputs[1].Value[0]
			}
		case "countb":
			node.Value[ind] = float64(ind)
		case "log":
			node.Value[ind] = math.Log(node.Inputs[0].Value[ind])
		case "exp":
			node.Value[ind] = math.Exp(node.Inputs[0].Value[ind])
		case "pow":
			node.Value[ind] = math.Pow(node.Inputs[0].Value[ind1], node.Inputs[1].Value[ind2])
		case "lag":
			if ind > 0 {
				node.Value[ind] = node.Inputs[0].Value[ind-1]
			} else {
				node.Value[ind] = node.Inputs[1].Value[0]
			}
		}

		// decrement indices
		ind1 -= deltas[0]

		if two {
			ind2 -= deltas[1]
		}
	}

	goNegative(node.Value, node.Neg)

	return nil
}

// evalConstant loads data which evaluates to a constant
func evalConstant(node *OpNode) bool {
	if val, e := strconv.ParseFloat(node.Expression, 64); e == nil {
		node.Value = make([]float64, 1)
		node.Value[0] = val
		goNegative(node.Value, node.Neg)

		return true
	}

	return false
}

// fromPipeline loads data which originates in the pipeline
func fromPipeline(node *OpNode, pipe Pipeline) error {
	field := node.Expression
	xLeftGD := pipe.Get(field)

	if xLeftGD == nil {
		return fmt.Errorf("%s not in pipeline", node.Expression)
	}

	node.Value = xLeftGD.Data.([]float64)

	if node.Neg {
		node.Value = make([]float64, pipe.Rows())
		copy(node.Value, xLeftGD.Data.([]float64))
		goNegative(node.Value, node.Neg)
	}

	return nil
}

// evalOps evaluates an operation
func evalOps(node *OpNode) error {
	if node.Inputs == nil || len(node.Inputs) != 2 {
		return fmt.Errorf("operations require two operands")
	}

	var deltas []int
	node.Value, deltas = getDeltas(node.Inputs)
	ind1, ind2 := 0, 0

	for ind := 0; ind < len(node.Value); ind++ {
		switch node.FunName {
		case "^":
			node.Value[ind] = math.Pow(node.Inputs[0].Value[ind1], node.Inputs[1].Value[ind2])
		case "&&":
			val := 0.0
			if node.Inputs[0].Value[ind1] > 0.0 && node.Inputs[1].Value[ind2] > 0.0 {
				val = 1
			}
			node.Value[ind] = val
		case "||":
			val := 0.0
			if node.Inputs[0].Value[ind1] > 0.0 || node.Inputs[1].Value[ind2] > 0.0 {
				val = 1
			}
			node.Value[ind] = val
		case ">":
			val := 0.0
			if node.Inputs[0].Value[ind1] > node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case ">=":
			val := 0.0
			if node.Inputs[0].Value[ind1] >= node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case "<":
			val := 0.0
			if node.Inputs[0].Value[ind1] < node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case "<=":
			val := 0.0
			if node.Inputs[0].Value[ind1] <= node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case "==":
			val := 0.0
			if node.Inputs[0].Value[ind1] == node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case "!=":
			val := 0.0
			if node.Inputs[0].Value[ind1] != node.Inputs[1].Value[ind2] {
				val = 1
			}
			node.Value[ind] = val
		case "+":
			node.Value[ind] = node.Inputs[0].Value[ind1] + node.Inputs[1].Value[ind2]
		case "-":
			node.Value[ind] = node.Inputs[0].Value[ind1] - node.Inputs[1].Value[ind2]
		case "*":
			node.Value[ind] = node.Inputs[0].Value[ind1] * node.Inputs[1].Value[ind2]
		case "/":
			node.Value[ind] = node.Inputs[0].Value[ind1] / node.Inputs[1].Value[ind2]
		}
		ind1 += deltas[0]
		ind2 += deltas[1]
	}
	goNegative(node.Value, node.Neg)

	return nil
}

// AddToPipe adds the Value slice in rootNode to pipe. The field will have name fieldName.
// To do this:
//  1. Create the *OpNode tree to evaluate the expression using Expr2Tree
//  2. Populate the values from a Pipeline using Evaluate.
//  3. Add the values to the Pipeline using AddToPipe
//
// Notes:
//   - AddToPipe can be within a CallBack to populate each new call to the database with the calculated fields.
//   - You can access the values after Evaluate without adding the field to the Pipeline from the Value element
//     of the root node.
func AddToPipe(rootNode *OpNode, fieldName string, pipe Pipeline) error {
	if rootNode.Value == nil {
		return fmt.Errorf("root node is nil")
	}

	if len(rootNode.Value) > 1 && len(rootNode.Value) != pipe.Rows() {
		return fmt.Errorf("AddtoPipe: exected length %d got length %d", pipe.Rows(), len(rootNode.Value))
	}

	xOut := rootNode.Value

	if len(xOut) == 1 && pipe.Rows() > 1 {
		xOut = make([]float64, pipe.Rows())
		for ind := 0; ind < len(xOut); ind++ {
			xOut[ind] = rootNode.Value[0]
		}
	}

	newRawField := NewRawCast(xOut, nil)

	return pipe.GData().AppendC(newRawField, fieldName, false, nil)
}

// Evaluate evaluates an expression parsed by Expr2Tree.
// The user calls Evaluate with the top node as returned by Expr2Tree
// To add a field to a pipeline:
//  1. Create the *OpNode tree to evaluate the expression using Expr2Tree
//  2. Populate the values from a Pipeline using Evaluate.
//  3. Add the values to the Pipeline using AddToPipe
//
// Note, you can access the values after Evaluate without adding the field to the Pipeline from the Value element
// of the root node.
func Evaluate(curNode *OpNode, pipe Pipeline) error {
	// recurse to evaluate from bottom up
	for ind := 0; ind < len(curNode.Inputs); ind++ {
		if e := Evaluate(curNode.Inputs[ind], pipe); e != nil {
			return e
		}
	}

	// check: are these operations +,-,*,/ ?
	if checkSlice(curNode.FunName, operations) {
		return evalOps(curNode)
	}

	// is this a function eval?
	if checkSlice(curNode.FunName, functions) || checkSlice(curNode.FunName, ifs) {
		return evalFunction(curNode)
	}

	// is it a constant?
	if evalConstant(curNode) {
		return nil
	}

	// must be a field from the pipeline then
	return fromPipeline(curNode, pipe)
}

// goNegative negates Value if Neg is true
func goNegative(x []float64, neg bool) {
	if !neg {
		return
	}

	for ind := 0; ind < len(x); ind++ {
		x[ind] = -x[ind]
	}
}

// matchedParen checks for mismatched parentheses
func matchedParen(expr string) error {
	if strings.Count(expr, "(") != strings.Count(expr, ")") {
		return fmt.Errorf("mismatched parentheses")
	}

	return nil
}