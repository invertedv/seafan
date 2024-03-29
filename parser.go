package seafan

import (
	_ "embed"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"

	grob "github.com/MetalBlueberry/go-plotly/graph_objects"

	"github.com/invertedv/utilities"
	"github.com/pkg/errors"

	"gonum.org/v1/gonum/optimize"
)

var (
	// FunctionsStr lists the functions that parser supports, the number and types of arguments, type of return
	//go:embed strings/functions.txt
	FunctionsStr string

	// Functions is a slice that describes all supported functions/operations
	Functions []FuncSpec

	Height = 1200.0
	Width  = 1200.0

	fig = &grob.Fig{}
)

const (
	// delimiter for strings below
	delim = "$"

	// logicals are disjunctions, conjunctions
	logicals = "&&$||"

	// Comparisons are comparison operators
	comparisons = ">$>=$<$<=$==$!="

	// these are separated out for order of precedence.
	arith1 = "+$-"

	arith2 = "*$/"

	arith3 = "^"

	// arithmetics is a list arithmetic operators
	arithmetics = arith1 + "$" + arith2 + "$" + arith3

	// operations is a list of supported operations
	operations = logicals + "$" + comparisons + "$" + arithmetics

	colors = "black,red,blue,green,yellow"
	mType  = "line,markers"
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
// Comparison operations with fields of type FRCat are permitted if the underlying data is type string or date.
// // Strings and dates are enclosed in a single quote ('). Date formats supported are: CCYYMMDD and MM/DD/CCYY.
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
//   - abs(<expr>) absolute value
//   - if(<test>, <true>, <false>), where the value <yes> is used if <condition> is greater than 0 and <false> o.w.
//   - row(<expr>) row number in pipeline. Row starts as 0 and is continuous.
//   - countAfter(<expr>), countBefore(<expr>) is the number of rows after (before) the current row.
//   - cumeAfter(<expr>), cumeBefore(<expr>,<missing>) is the cumulative sum of <expr> after (before) the current row (included)
//   - prodAfter(<expr>), prodBefore(<expr>,<missing>) is the cumulative product of <expr> after (before) the current row (included)
//     and <missing> is used for the last (first) element.
//   - index(<expr>,<index>) returns <expr> in the order of <index>
//   - cat(<expr>) converts <expr> to a categorical field. Only applicable to continuous fields.
//   - toDate(<expr>) converts a string field to a date
//   - toString(<expr>) converts <expr> to string
//   - toFloatSP(<expr>) converts <expr> to float32
//   - toFloatDP(<expr>) converts <expr> to float64
//   - toInt(<expr>) converts <expr> to int.  Same as cat().
//   - dateAdd(<date>,<months>) adds <months> to the date, <date>
//   - toLastDayOfMonth(<date>)  moves the date to the last day of the month
//   - toFirstDayOfMonth(<date>) moves the date to the first day of the month
//   - year(<date>) returns the year
//   - month(<date>) returns the month (1-12)
//   - day(<date>) returns the day of the month (1-lastDayOfMonth)
//   - dateDiff(<data1>,<date2>,unit) returns date1-date2 units can be 'hour', 'day', 'month' or 'year'
//   - nowDate() returns current date
//   - nowTime() returns current time as a string
//   - substr(<string>,<start>,<length>) substring
//   - strPos(<string>,<target>) first position of <target> in <string>. -1 if does not occur.
//   - strCount(<string>,<target>) number of times <target> occurs in <string>
//   - strLen(<string>) length of string
//   - trunc(<expr>)  truncate to int
//   - exist(x,y) if x exists, returns x. If x does not exist, returns y.
//
// The values in <...> can be any expression.  The functions prodAfter, prodBefore, cumAfter,cumBefore,
// countAfter, countBefore do NOT include the current row.
//
// Available summary-level functions are:
//   - mean(<expr>)
//   - count(<expr>)
//   - sum(<expr>)
//   - max(<expr>)
//   - min(<expr>)
//   - sse(<y>,<yhat>) returns the sum of squared error of y-yhat
//   - mad(<y>,<yhat>) returns the sum of the absolute value of y-yhat
//   - r2(<y>,<yhat>) returns the r-square of estimating y with yhat
//   - npv(<discount rate>, <cash flows>).  Find the NPV of the cash flows at discount rate. If disount rate
//     is a slice, then the ith month's cashflows are discounted for i months at the ith discount rate.
//   - irr(<cost>,<cash flows>).  Find the IRR of an initial outlay of <cost> (a positive value!), yielding cash flows
//     (The first cash flow gets discounted one period). irr returns 0 if there's no solution.
//   - print(<expr>,<rows>) print <rows> of the <expr>.  If <rows>=0, print entire slice.
//   - printIf(<expr>,<rows>,<cond>) if condition evaluates to a value > 0, execute print(<expr>,<rows>)
//   - histogram(<x>,<color>, <normalization>).  Creates a histogram. normalization is one of: percent, count, density
//   - plotLine(<x>,<markerType>, <color>)
//   - plotXY(<x>,<y>,<markerType>, <color>)
//   - setPlotDim(<width>,<height>), <width>, <height> are in pixels
//   - render(<file>,<title>,<x label>,<y label>)
//   - newPlot()
//
// Comparisons
//   - ==, !=, >,>=, <, <=
//
// Logical operators are supported:
//   - && for "and"
//   - || for "or"
//
// Logical operators resolve to 0 or 1.
type OpNode struct {
	Expression string    // expression this node implements
	Raw        *Raw      // node Value
	Func       *FuncSpec // details of the function required to evaulate this node.
	Role       FRole     // FRole to use when adding this node to a Pipeline
	Neg        bool      // negate result when populating Value
	Inputs     []*OpNode // Inputs to node calculation
	stet       bool      // if stet then Value is not updated (used by Loop)
}

// FuncSpec stores the details about a function call.
type FuncSpec struct {
	Name   string         // The name of the function/operation.
	Return reflect.Kind   // The type of the return.  This will either be float64 or any.
	Args   []reflect.Kind // The types of the inputs to the function.
	Level  rune           // 'S' if the function is summary-level (1 element) or 'R' if it is row-level.
}

// loadFunctions loads the slice of FuncSpec that is all the defined functions the parser supports.
func loadFunctions() {
	funcs := strings.Split(strings.ReplaceAll(FunctionsStr, "\n", ""), "$")
	for _, f := range funcs {
		fdetail := strings.Split(f, ",")
		fSpec := FuncSpec{Name: fdetail[0],
			Return: utilities.String2Kind(fdetail[1]),
			Level:  rune(fdetail[2][0]),
		}

		for ind := 3; ind < len(fdetail); ind++ {
			if fdetail[ind] != "" {
				fSpec.Args = append(fSpec.Args, utilities.String2Kind(fdetail[ind]))
			}
		}
		Functions = append(Functions, fSpec)
	}
}

// Expr2Tree builds the OpNode tree that is a binary tree representation of an expression.
// The process to add a field to a Pipeline is:
//  1. Create the *OpNode tree using Expr2Tree to evaluate the expression
//  2. Populate the values from a Pipeline using Evaluate.
//  3. Add the values to the Pipeline using AddToPipe.
//
// Note, you can access the values after Evaluate without adding the field to the Pipeline from the Raw field
// of the root node.
//
// The expression can include:
//   - arithmetic operators: +, -, *, /
//   - exponentation: ^
//   - functions
//   - logicals: &&, ||.  These evaluate to 0 or 1.
//   - if statements: if(condition, value if true, value if false). The true value is applied if the condition evaluates
//     to a positive value.
//   - parentheses
func Expr2Tree(curNode *OpNode) error {
	// Load the global slice of functions if they are not
	if Functions == nil {
		loadFunctions()
	}

	curNode.Expression = utilities.ReplaceSmart(curNode.Expression, " ", "", "'")

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

	curNode.Func, curNode.Role = getFuncSpec(op)
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

// getFuncSpec returns the FuncSpec for the function/operation op
// FRole is the default role for the function
func getFuncSpec(op string) (*FuncSpec, FRole) {
	for _, fSpec := range Functions {
		if op == fSpec.Name {
			var role FRole
			switch fSpec.Return {
			case reflect.String, reflect.Struct:
				role = FRCat
			case reflect.Interface:
				role = FREither
			default:
				role = FRCts
			}

			return &fSpec, role
		}
	}

	return nil, FREither
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

// find the first needle that is not within parens.  Ignore the first character--that cannot be a true operator.
// Needles string uses delim to separate the needles
func searchOp(expr, needles string) (op string, args []string) {
	if expr == "" {
		return "", []string{expr}
	}

	ignore := 0
	ignoreQ := false // single quote
	for indx := 0; indx < len(expr)-1; indx++ {
		// needles can be 1 or 2 characters wide
		ch := expr[indx : indx+1]
		ch2 := expr[indx : indx+2]
		switch ch {
		case "(":
			ignore++
		case ")":
			ignore--
		case "'":
			ignoreQ = !ignoreQ
		default:
			if ignore == 0 && !ignoreQ && indx > 0 {
				// check 2-character needles first
				if utilities.Has(ch2, delim, needles) {
					return ch2, []string{expr[0:indx], expr[indx+2:]}
				}

				if utilities.Has(ch, delim, needles) {
					return ch, []string{expr[0:indx], expr[indx+1:]}
				}
			}
		}
	}

	return "", nil
}

// getArgs breaks up function arguments into elements of a slice
func getArgs(inner string) (pieces []string) {
	// no arguments
	if inner == "" {
		return
	}

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
	f := expr[0:indx]
	for ind := 0; ind < indx; ind++ {
		if !strings.Contains("abcdefghijklmnopqrstuvwxyz", strings.ToLower(f[ind:ind+1])) {
			return "", nil, nil
		}
	}

	fSpec, _ := getFuncSpec(f)
	// Is this a known function?
	if fSpec == nil {
		return f, nil, fmt.Errorf("unknown function: %s", f)
	}

	// get arguments
	args = getArgs(inner)

	if fSpec.Args != nil && len(fSpec.Args) != len(args) {
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

	// If this is a function, we will create a node just to calculate it and then recurse to get the arguments
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

	op, args = searchOp(expr, arith3)
	if args != nil {
		return op, args, nil
	}

	return "", []string{expr}, nil
}

// ifCond evaluates an "if" condition.
func ifCond(node *OpNode) error {
	var deltas []int
	node.Raw, deltas = getDeltas(node)

	// do some checking on lengths
	indT, indF := 0, 0
	for indRes := 0; indRes < node.Raw.Len(); indRes++ {
		x := node.Inputs[2].Raw.Data[indF]
		if node.Inputs[0].Raw.Data[indRes].(float64) > 0.0 {
			x = node.Inputs[1].Raw.Data[indT]
		}
		node.Raw.Data[indRes] = x
		indT += deltas[1]
		indF += deltas[2]
	}

	return nil
}

// getDeltas returns an array for the results and a slice of increments for moving through the Inputs
func getDeltas(node *OpNode) (x *Raw, deltas []int) {
	if node.Inputs == nil {
		return nil, nil
	}

	n := 1
	for ind := 0; ind < len(node.Inputs); ind++ {
		if node.Inputs[ind].Raw == nil {
			return nil, nil
		}

		d := 0
		nx := node.Inputs[ind].Raw.Len()
		if nx > 1 {
			d = 1
			if nx > n {
				n = nx
			}
		}

		deltas = append(deltas, d)
	}

	return AllocRaw(n, node.Func.Return), deltas
}

// npv finds NPV when the discount rate is a constant. The first cashflow has a discount factor of 1.0
func npv(discount, cashflows *Raw) (pv float64) {
	r := 1.0 / (1.0 + discount.Data[0].(float64))
	totalD := 1.0
	for ind := 0; ind < cashflows.Len(); ind++ {
		if discount.Len() == 1 {
			if ind > 0 {
				totalD *= r
			}
		} else {
			totalD = math.Pow(1.0/(1.0+discount.Data[ind].(float64)), float64(ind))
		}

		pv += cashflows.Data[ind].(float64) * totalD
	}

	return pv
}

// print the slice (and return 1)
func printer(toPrint *Raw, name string, numPrint any) (*Raw, error) {
	asInt32, e := utilities.Any2Int32(numPrint)
	if e != nil {
		return nil, e
	}

	if asInt32 == nil {
		return nil, fmt.Errorf("cannot convert # rows to print to int32")
	}

	num2Print := int(*asInt32)

	if num2Print == 0 {
		num2Print = toPrint.Len()
	}

	if num2Print < 0 {
		return nil, fmt.Errorf("negative # rows to print")
	}

	num2Print = utilities.MinInt(num2Print, toPrint.Len())

	fmt.Println(name)
	for ind := 0; ind < num2Print; ind++ {
		fmt.Printf("%d: %v\n", ind, toPrint.Data[ind])
	}

	return NewRaw([]any{1.0}, nil), nil
}

func printIf(toPrint *Raw, name string, numPrint, cond any) (*Raw, error) {
	asFlt, e := utilities.Any2Float64(cond)
	if e != nil {
		return nil, errors.WithMessage(e, "printIf")
	}

	if *asFlt <= 0.0 {
		return NewRaw([]any{0.0}, nil), nil
	}

	return printer(toPrint, name, numPrint)
}

// irr finds the internal rate of return of the cashflows against the initial outlay of cost.
// guess0 is the initial guess to the optimizer.
func irr(cost, guess0 float64, cashflows *Raw) (float64, error) {
	const (
		tolValue = 1e-4
		maxIter  = 40
	)
	var optimal *optimize.Result

	irrValue := []float64{guess0}

	obj := func(irrValue []float64) float64 {
		irrv := NewRaw([]any{irrValue[0]}, nil)
		resid := npv(irrv, cashflows) - cost
		return resid * resid
	}
	problem := optimize.Problem{Func: obj}

	// optimize
	settings := &optimize.Settings{
		InitValues:        nil,
		GradientThreshold: 0,
		Converger:         nil,
		MajorIterations:   maxIter,
		Runtime:           0,
		FuncEvaluations:   0,
		GradEvaluations:   0,
		HessEvaluations:   0,
		Recorder:          nil,
		Concurrent:        12,
	}

	optimal, _ = optimize.Minimize(problem, irrValue, settings, &optimize.NelderMead{})
	if optimal == nil {
		return 0, fmt.Errorf("irr failed")
	}

	pv := npv(NewRawCast(optimal.X, nil), cashflows)
	if math.Abs(pv-cost) > math.Abs(tolValue*cost) {
		return 0, fmt.Errorf("irr failed")
	}

	return optimal.X[0], nil
}

// sseMAD returns the SSE of y to yhat (op="sse") and the MAD (actually, the sum) o.w.
func sseMAD(y, yhat *Raw, op string) float64 {
	resid := make([]float64, y.Len())
	for ind, r := range y.Data {
		resid[ind] = r.(float64) - yhat.Data[ind].(float64)
	}

	val := 0.0
	if op == "sse" {
		for ind := 0; ind < len(resid); ind++ {
			val += resid[ind] * resid[ind]
		}
	} else {
		for ind := 0; ind < len(resid); ind++ {
			val += math.Abs(resid[ind])
		}
	}

	return val
}

// generate a slice that runs from start to end
func ranger(start, end any) (*Raw, error) {
	var (
		begPtr, finishPtr *int32
		e                 error
	)

	if begPtr, e = utilities.Any2Int32(start); e != nil {
		return nil, errors.WithMessage(e, "ranger")
	}

	if finishPtr, e = utilities.Any2Int32(end); e != nil {
		return nil, errors.WithMessage(e, "ranger")
	}

	beg := *begPtr
	finish := *finishPtr

	if beg == finish {
		return nil, fmt.Errorf("empty range")
	}

	var data []any

	var delta int32 = 1
	if finish < beg {
		delta = -1
	}

	ind, ok := beg, true
	for ok {
		data = append(data, ind)
		ind += delta

		if ind == finish {
			ok = false
		}
	}

	rng := NewRaw(data, nil)

	return rng, nil
}

// EvalSFunction evaluates a summary function. A summary function returns a single value.
func EvalSFunction(node *OpNode) error {
	const irrGuess = 0.005

	var e error
	var result *Raw

	switch node.Func.Name {
	case "print":
		result, e = printer(node.Inputs[0].Raw, node.Inputs[0].Expression, node.Inputs[1].Raw.Data[0])
	case "printIf":
		result, e = printIf(node.Inputs[0].Raw, node.Inputs[0].Expression, node.Inputs[1].Raw.Data[0], node.Inputs[2].Raw.Data[0])
	case "plotXY":
		result, e = plotXY(node.Inputs[0].Raw, node.Inputs[1].Raw, node.Inputs[2].Raw, node.Inputs[3].Raw)
	case "plotLine":
		result, e = plotLine(node.Inputs[0].Raw, node.Inputs[1].Raw, node.Inputs[2].Raw)
	case "histogram":
		result, e = histogram(node.Inputs[0].Raw, node.Inputs[1].Raw, node.Inputs[2].Raw)
	case "setPlotDim":
		result, e = setPlotDim(node.Inputs[0].Raw, node.Inputs[1].Raw)
	case "newPlot":
		result = newPlot()
	case "render":
		result, e = render(node.Inputs[0].Raw, node.Inputs[1].Raw, node.Inputs[2].Raw, node.Inputs[3].Raw)
	case "sum":
		result, e = node.Inputs[0].Raw.Sum()
	case "max":
		result, e = node.Inputs[0].Raw.Max()
	case "min":
		result, e = node.Inputs[0].Raw.Min()
	case "mean":
		result, e = node.Inputs[0].Raw.Mean()
	case "std":
		result, e = node.Inputs[0].Raw.Std()
	case "count":
		result = NewRaw([]any{int32(node.Inputs[0].Raw.Len())}, nil)
	case "npv":
		result = NewRaw([]any{npv(node.Inputs[0].Raw, node.Inputs[1].Raw)}, nil)
	case "irr":
		irrValue, _ := irr(node.Inputs[0].Raw.Data[0].(float64), irrGuess, node.Inputs[1].Raw)
		result = NewRaw([]any{irrValue}, nil)
	case "sse", "mad":
		result = NewRaw([]any{sseMAD(node.Inputs[0].Raw, node.Inputs[1].Raw, "sse")}, nil)
	case "r2":
		num := sseMAD(node.Inputs[0].Raw, node.Inputs[1].Raw, "sse")

		denR, ex := node.Inputs[0].Raw.Std()
		if ex != nil {
			return ex
		}

		den := denR.Data[0].(float64)
		den = den * den * (float64(node.Inputs[0].Raw.Len() - 1))
		result = NewRaw([]any{1.0 - num/den}, nil)
		//	case "corr":
		//		node.Value = []float64{stat.Correlation(node.Inputs[0].Value, node.Inputs[1].Value, nil)}
	default:
		return fmt.Errorf("unknown function: %s", node.Func.Name)
	}

	if e != nil {
		return e
	}
	node.Raw = NewRaw([]any{result.Data[0]}, nil)
	goNegative(node.Raw, node.Neg)

	return nil
}

// toLastDayOfMonth moves the date to the last day of the month
func toLastDayOfMonth(node *OpNode) error {
	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to toLastDayOfMonth isn't a date")
	}

	n := node.Inputs[0].Raw.Len()
	dates := make([]any, n)

	for ind := 0; ind < n; ind++ {
		dt, ok := node.Inputs[0].Raw.Data[ind].(time.Time)
		if !ok {
			return fmt.Errorf("arg 1 to dateadd isn't a date")
		}

		dates[ind] = utilities.ToLastDay(dt)
	}

	node.Raw = NewRaw(dates, nil)

	return nil
}

// toLastDayOfMonth moves the date to the last day of the month
func toFirstDayOfMonth(node *OpNode) error {
	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to toFirstDayOfMonth isn't a date")
	}

	n := node.Inputs[0].Raw.Len()
	dates := make([]any, n)

	for ind := 0; ind < n; ind++ {
		dt, ok := node.Inputs[0].Raw.Data[ind].(time.Time)
		if !ok {
			return fmt.Errorf("arg 1 to dateadd isn't a date")
		}

		dates[ind] = time.Date(dt.Year(), dt.Month(), 1, 0, 0, 0, 0, time.UTC)
	}

	node.Raw = NewRaw(dates, nil)

	return nil
}

// monthYearDay returns the month/year/day from a date
func monthYearDay(node *OpNode, part string) error {
	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to month isn't a date")
	}

	n := node.Inputs[0].Raw.Len()
	months := make([]any, n)

	for ind := 0; ind < n; ind++ {
		dt, ok := node.Inputs[0].Raw.Data[ind].(time.Time)
		if !ok {
			return fmt.Errorf("arg 1 to dateadd isn't a date")
		}

		switch part {
		case "year":
			months[ind] = int32(dt.Year())
		case "month":
			months[ind] = int32(dt.Month())
		case "day":
			months[ind] = int32(dt.Day())
		}
	}

	node.Raw = NewRaw(months, nil)

	return nil
}

// dateDiff finds the difference between two dates in hours, days, months or years
func dateDiff(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to dateadd isn't a date")
	}

	n := utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len())
	dates := make([]any, n)
	ind1, ind2 := 0, 0

	for ind := 0; ind < n; ind++ {
		dt1, ok := node.Inputs[0].Raw.Data[ind1].(time.Time)
		if !ok {
			return fmt.Errorf("arg 1 to datesub isn't a date")
		}

		dt2, ok := node.Inputs[1].Raw.Data[ind2].(time.Time)
		if !ok {
			return fmt.Errorf("arg 2 to datesub isn't a date")
		}

		unit, ok := node.Inputs[2].Raw.Data[0].(string)
		if !ok {
			return fmt.Errorf("arg 3 to datesub isn't a string")
		}

		y1, m1, d1 := dt1.Date()
		y2, m2, d2 := dt2.Date()
		var val int32

		switch unit {
		case "hour":
			val = int32(dt1.Sub(dt2).Hours())
		case "day":
			ta := time.Date(y1, m1, d1, 0, 0, 0, 0, time.UTC)
			tb := time.Date(y2, m2, d2, 0, 0, 0, 0, time.UTC)
			val = int32(ta.Sub(tb) / 24)
		case "month":
			val = int32(12*y1 + int(m1) - (12*(y2) + int(m2)))
		case "year":
			val = int32(y1 - y2)
		}

		dates[ind] = val
		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	node.Raw = NewRaw(dates, nil)

	return nil
}

// substr finds a substring
func substr(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to substr is missing")
	}

	n := utilities.MaxInt(utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len()), node.Inputs[2].Raw.Len())
	stringsX := make([]any, n)
	ind1, ind2, ind3 := 0, 0, 0

	for ind := 0; ind < n; ind++ {
		str, ok := node.Inputs[0].Raw.Data[ind1].(string)
		if !ok {
			return fmt.Errorf("arg 1 to substr isn't a string")
		}

		var (
			start, end int32
			x          any
			err        error
		)

		if x, err = utilities.Any2Kind(node.Inputs[1].Raw.Data[ind2], reflect.Int32); err != nil {
			return fmt.Errorf("arg 2 to substr isn't an int")
		}

		start = x.(int32) - 1

		if x, err = utilities.Any2Kind(node.Inputs[2].Raw.Data[ind3], reflect.Int32); err != nil {
			return fmt.Errorf("arg 3 to substr isn't an int")
		}
		end = start + x.(int32)
		if end >= int32(len(str)) {
			end = int32(len(str))
		}

		stringsX[ind] = str[start:end]
		ind1 += deltas[0]
		ind2 += deltas[1]
		ind3 += deltas[2]
	}

	node.Raw = NewRaw(stringsX, nil)

	return nil
}

// strCount counts the occurences of arg2 in arg1
func strCount(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to strPos is missing")
	}

	n := utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len())
	locs := make([]any, n)
	ind1, ind2 := 0, 0

	for ind := 0; ind < n; ind++ {
		str, ok := node.Inputs[0].Raw.Data[ind1].(string)
		if !ok {
			return fmt.Errorf("arg 1 to substr isn't a string")
		}

		var (
			look string
		)

		if look, ok = node.Inputs[1].Raw.Data[ind2].(string); !ok {
			return fmt.Errorf("arg 2 to strPos isn't a string")
		}

		skip, cnt := len(look), 0
		for loc := strings.Index(str, look); loc >= 0; loc = strings.Index(str, look) {
			if loc < 0 {
				break
			}

			cnt++

			if loc+skip >= len(str) {
				break
			}

			str = str[loc+skip:]
		}

		locs[ind] = float64(cnt)
		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	node.Raw = NewRaw(locs, nil)

	return nil
}

// strLen returns the length of a string
func strLen(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to strPos is missing")
	}

	n := node.Inputs[0].Raw.Len()
	locs := make([]any, n)
	ind1 := 0

	for ind := 0; ind < n; ind++ {
		str, ok := node.Inputs[0].Raw.Data[ind1].(string)
		if !ok {
			return fmt.Errorf("arg 1 to substr isn't a string")
		}

		locs[ind] = float64(len(str))
		ind1 += deltas[0]
	}

	node.Raw = NewRaw(locs, nil)

	return nil
}

// abs takes the absolute value
func abs(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to abs is missing")
	}

	n := node.Inputs[0].Raw.Len()
	fabs := make([]any, n)
	ind1 := 0

	for ind := 0; ind < n; ind++ {
		x, err := utilities.Any2Float64(node.Inputs[0].Raw.Data[ind1])
		if err != nil {
			return err
		}

		fabs[ind] = math.Abs(*x)
		ind1 += deltas[0]
	}

	node.Raw = NewRaw(fabs, nil)

	return nil
}

// strPos returns the index of the first occurence of arg2 in arg1, -1 if not there
func strPos(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to strCount is missing")
	}

	n := utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len())
	locs := make([]any, n)
	ind1, ind2 := 0, 0

	for ind := 0; ind < n; ind++ {
		str, ok := node.Inputs[0].Raw.Data[ind1].(string)
		if !ok {
			return fmt.Errorf("arg 1 to strCount isn't a string")
		}

		var (
			look string
			loc  int
		)

		if look, ok = node.Inputs[1].Raw.Data[ind2].(string); !ok {
			return fmt.Errorf("arg 2 to strPos isn't a string")
		}

		loc = strings.Index(str, look)
		if loc = strings.Index(str, look); loc >= 0 {
			loc++
		}

		locs[ind] = float64(loc)

		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	node.Raw = NewRaw(locs, nil)

	return nil
}

// dateAddMonths adds months to a date field
func dateAddMonths(node *OpNode) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to dateadd isn't a date")
	}

	n := utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len())
	dates := make([]any, n)
	ind1, ind2 := 0, 0

	for ind := 0; ind < n; ind++ {
		dt, ok := node.Inputs[0].Raw.Data[ind1].(time.Time)
		if !ok {
			return fmt.Errorf("arg 1 to dateadd isn't a date")
		}

		var (
			param any
			e     error
		)

		if param, e = utilities.Any2Kind(node.Inputs[1].Raw.Data[ind2], reflect.Int32); e != nil {
			return errors.WithMessage(e, "dateAddMonths")
		}

		dates[ind] = dt.AddDate(0, int(param.(int32)), 0)

		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	node.Raw = NewRaw(dates, nil)

	return nil
}

// maxmin2 takes the element-wise max/min of the two inputs
func maxmin2(node *OpNode, maxmin string) error {
	var deltas []int

	_, deltas = getDeltas(node)

	if node.Inputs[0].Raw == nil {
		return fmt.Errorf("arg 1 to dateadd isn't a date")
	}

	n := utilities.MaxInt(node.Inputs[0].Raw.Len(), node.Inputs[1].Raw.Len())
	mm := make([]any, n)
	ind1, ind2 := 0, 0

	for ind := 0; ind < n; ind++ {
		var e error

		val := node.Inputs[0].Raw.Data[ind1]
		t := reflect.TypeOf(val).Kind()

		var y any
		if y, e = utilities.Any2Kind(node.Inputs[1].Raw.Data[ind2], t); e != nil {
			return fmt.Errorf("cannot coerce values to same type, max2/min2")
		}

		switch x := node.Inputs[0].Raw.Data[ind1].(type) {
		case int32:
			if (maxmin == "max" && y.(int32) > x) || (maxmin == "min" && y.(int32) < x) {
				val = y
			}
		case int64:
			if (maxmin == "max" && y.(int64) > x) || (maxmin == "min" && y.(int64) < x) {
				val = y
			}
		case float32:
			if (maxmin == "max" && y.(float32) > x) || (maxmin == "min" && y.(float32) < x) {
				val = y
			}
		case float64:
			if (maxmin == "max" && y.(float64) > x) || (maxmin == "min" && y.(float64) < x) {
				val = y
			}
		case string:
			if (maxmin == "max" && y.(string) > x) || (maxmin == "min" && y.(string) < x) {
				val = y
			}
		case time.Time:
			if (maxmin == "max" && y.(time.Time).Sub(x) > 0) || (maxmin == "min" && y.(time.Time).Sub(x) < 0) {
				val = y
			}
		}

		mm[ind] = val

		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	node.Raw = NewRaw(mm, nil)

	return nil
}

// toWhatever attempts to convert the values in node to kind
func toWhatever(node *OpNode, kind reflect.Kind) error {
	xIn := node.Inputs[0].Raw.Data
	n := len(xIn)
	xOut := make([]any, n)

	for ind := 0; ind < n; ind++ {
		val, e := utilities.Any2Kind(xIn[ind], kind)

		if e != nil {
			return fmt.Errorf("conversion to %v failed", kind)
		}

		xOut[ind] = val
	}

	node.Raw = NewRaw(xOut, nil)

	return nil
}

// nowDate puts the current date into node
func nowDate(node *OpNode) error {
	xOut := make([]any, 1)
	xOut[0] = time.Now()
	node.Raw = NewRaw(xOut, nil)

	return nil
}

// nowTime puts the current date/time as a string into node
func nowTime(node *OpNode) error {
	xOut := make([]any, 1)
	now := time.Now()
	xOut[0] = fmt.Sprintf("%d:%d:%d", now.Hour(), now.Minute(), now.Second())
	node.Raw = NewRaw(xOut, nil)

	return nil
}

// evalFunction evaluates a function call
func evalFunction(node *OpNode) error {
	if e := consistent(node); e != nil {
		return e
	}
	var err error
	gotOne := true
	// special cases
	switch node.Func.Name {
	case "if":
		err = ifCond(node)
	case "dateAdd":
		err = dateAddMonths(node)
	case "dateDiff":
		err = dateDiff(node)
	case "nowDate":
		err = nowDate(node)
	case "nowTime":
		err = nowTime(node)
	case "toLastDayOfMonth":
		err = toLastDayOfMonth(node)
	case "toFirstDayOfMonth":
		err = toFirstDayOfMonth(node)
	case "month":
		err = monthYearDay(node, "month")
	case "year":
		err = monthYearDay(node, "year")
	case "day":
		err = monthYearDay(node, "day")
	case "maxE":
		err = maxmin2(node, "max")
	case "minE":
		err = maxmin2(node, "min")
	case "substr":
		err = substr(node)
	case "strPos":
		err = strPos(node)
	case "strCount":
		err = strCount(node)
	case "strLen":
		err = strLen(node)
	case "toDate":
		err = toWhatever(node, reflect.Struct)
	case "toString":
		err = toWhatever(node, reflect.String)
	case "toFloatDP":
		node.Role = FRCts
		err = toWhatever(node, reflect.Float64)
	case "toFloatSP":
		node.Role = FRCts
		err = toWhatever(node, reflect.Float32)
	case "toInt", "cat":
		node.Role = FRCat
		err = toWhatever(node, reflect.Int32)
	case "abs":
		err = abs(node)
	default:
		gotOne = false
	}

	if err != nil {
		return err
	}

	if gotOne {
		goNegative(node.Raw, node.Neg)
		return nil
	}

	if node.Func != nil && node.Func.Level == 'S' {
		if e := EvalSFunction(node); e != nil {
			return e
		}

		return nil
	}

	switch node.Func.Name {
	case "exist":
		node.Raw, err = node.Inputs[0].Raw, nil
	case "cumeAfter":
		node.Raw, err = node.Inputs[0].Raw.CumeAfter("sum")
	case "prodAfter":
		node.Raw, err = node.Inputs[0].Raw.CumeAfter("product")
	case "countAfter":
		node.Raw, err = node.Inputs[0].Raw.CumeAfter("count")
	case "cumeBefore":
		node.Raw, err = node.Inputs[0].Raw.CumeBefore("sum")
	case "prodBefore":
		node.Raw, err = node.Inputs[0].Raw.CumeBefore("product")
	case "countBefore":
		node.Raw, err = node.Inputs[0].Raw.CumeBefore("count")
	case "row":
		node.Raw, err = node.Inputs[0].Raw.CumeBefore("count")
		for ind, x := range node.Raw.Data {
			node.Raw.Data[ind] = x.(float64) - 1
		}
	case "lag":
		node.Raw, err = node.Inputs[0].Raw.Lag(node.Inputs[1].Raw.Data[0])
	case "pow":
		node.Raw, err = node.Inputs[0].Raw.Pow(node.Inputs[1].Raw)
	case "range":
		node.Raw, err = ranger(node.Inputs[0].Raw.Data[0], node.Inputs[1].Raw.Data[0])
	case "index":
		node.Raw, err = node.Inputs[0].Raw.Index(node.Inputs[1].Raw)
	case "exp":
		node.Raw, err = node.Inputs[0].Raw.Exp()
	case "log":
		node.Raw, err = node.Inputs[0].Raw.Log()
	default:
		return fmt.Errorf("unknown function %s", node.Func.Name)
	}

	if err != nil {
		return err
	}

	goNegative(node.Raw, node.Neg)

	return nil
}

// evalConstant loads data which evaluates to a constant
func evalConstant(node *OpNode) bool {
	if val, e := strconv.ParseFloat(node.Expression, 64); e == nil {
		node.Raw = AllocRaw(1, reflect.Float64)
		node.Raw.Data[0] = val
		node.Role = FRCts

		goNegative(node.Raw, node.Neg)

		return true
	}

	if strings.Contains(node.Expression, "'") {
		// strip single quote
		node.Raw = NewRaw([]any{strings.ReplaceAll(node.Expression, "'", "")}, nil)
		node.Role = FRCat

		return true
	}

	return false
}

// fromPipeline loads data which originates in the pipeline
func fromPipeline(node *OpNode, pipe Pipeline) error {
	field := node.Expression

	var e error
	node.Raw, e = pipe.GData().GetRaw(field)
	if e != nil {
		return fmt.Errorf("%s not in pipeline", node.Expression)
	}

	// if node.Neg then need to copy data into node.Raw so it doesn't affect the data in the Pipeline
	if node.Neg {
		xOut := make([]any, node.Raw.Len())
		copy(xOut, node.Raw.Data)
		node.Raw = NewRaw(xOut, nil)
		goNegative(node.Raw, node.Neg)
	}

	ft := pipe.GetFType(field)
	if ft.Role == FROneHot || ft.Role == FREmbed {
		return fmt.Errorf("cannot operate on onehot or embedded fields")
	}

	node.Role = ft.Role

	return nil
}

// evalOpsCat evaluates operations (inequalities) for FRCat (string, date) fields
func evalOpsCat(node *OpNode) error {
	var deltas []int
	node.Raw, deltas = getDeltas(node)
	ind1, ind2 := 0, 0

	for ind := 0; ind < node.Raw.Len(); ind++ {
		// check same type...
		node.Raw.Data[ind] = float64(0)
		test, e := utilities.Comparer(node.Inputs[0].Raw.Data[ind1], node.Inputs[1].Raw.Data[ind2], node.Func.Name)
		if e != nil {
			return e
		}

		if test {
			node.Raw.Data[ind] = float64(1)
		}

		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	return nil
}

// consistent checks that the Inputs are consistent with what's needed as specified in node.Func.args
func consistent(node *OpNode) error {
	if node.Func == nil {
		return nil
	}

	if len(node.Inputs) != len(node.Func.Args) {
		return fmt.Errorf("argument count mismatch")
	}

	for ind, arg := range node.Func.Args {
		switch arg {
		case reflect.Float64:
			switch node.Inputs[ind].Raw.Kind {
			case reflect.Struct, reflect.String:
				return fmt.Errorf("argument type mismatch, function %s", node.Func.Name)
			}
		case reflect.Struct:
			if node.Inputs[ind].Raw.Kind != reflect.Struct {
				return fmt.Errorf("argument type mismatch, function %s", node.Func.Name)
			}
		}
	}

	return nil
}

// evalOps evaluates an operation
func evalOps(node *OpNode) error {
	if node.Inputs == nil || len(node.Inputs) != 2 {
		return fmt.Errorf("operations require two operands")
	}

	if e := consistent(node); e != nil {
		return e
	}

	// interface?
	if node.Func.Return == reflect.String || node.Func.Return == reflect.Struct || node.Func.Return == reflect.Interface {
		return evalOpsCat(node)
	}

	var deltas []int
	node.Raw, deltas = getDeltas(node)
	ind1, ind2 := 0, 0

	for ind := 0; ind < node.Raw.Len(); ind++ {
		if ind1 >= node.Inputs[0].Raw.Len() || ind2 >= node.Inputs[1].Raw.Len() {
			return fmt.Errorf("slices not same length")
		}
		var (
			x0, x1 any
			e0, e1 error
		)
		x0, e0 = utilities.Any2Kind(node.Inputs[0].Raw.Data[ind1], reflect.Float64)
		x1, e1 = utilities.Any2Kind(node.Inputs[1].Raw.Data[ind2], reflect.Float64)
		if e0 != nil || e1 != nil {
			return fmt.Errorf("cannot convert")
		}

		switch node.Func.Name {
		case "^":
			node.Raw.Data[ind] = math.Pow(x0.(float64), x1.(float64))
		case "&&":
			val := 0.0

			if x0.(float64) > 0.0 && x1.(float64) > 0.0 {
				val = 1
			}

			node.Raw.Data[ind] = val
		case "||":
			val := 0.0

			if x0.(float64) > 0.0 || x1.(float64) > 0.0 {
				val = 1
			}

			node.Raw.Data[ind] = val
		case ">", ">=", "==", "!=", "<", "<=":
			node.Raw.Data[ind] = float64(0)
			test, _ := utilities.Comparer(x0, x1, node.Func.Name)
			if test {
				node.Raw.Data[ind] = float64(1)
			}
		case "+":
			node.Raw.Data[ind] = x0.(float64) + x1.(float64)
		case "*":
			node.Raw.Data[ind] = x0.(float64) * x1.(float64)
		case "/":
			if x1.(float64) == 0.0 {
				return fmt.Errorf("divide by zero")
			}

			node.Raw.Data[ind] = x0.(float64) / x1.(float64)
		}

		ind1 += deltas[0]
		ind2 += deltas[1]
	}

	goNegative(node.Raw, node.Neg)

	return nil
}

// Evaluate evaluates an expression parsed by Expr2Tree.
// The user calls Evaluate with the top node as returned by Expr2Tree
// To add a field to a pipeline:
//  1. Create the *OpNode tree to evaluate the expression using Expr2Tree
//  2. Populate the values from a Pipeline using Evaluate.
//  3. Add the values to the Pipeline using AddToPipe
//
// Note, you can access the values after Evaluate without adding the field to the Pipeline from the *Raw item
// of the root node.
func Evaluate(curNode *OpNode, pipe Pipeline) error {
	// recurse to evaluate from bottom up
	for ind := 0; ind < len(curNode.Inputs); ind++ {

		e := Evaluate(curNode.Inputs[ind], pipe)

		// Super special case: "exist" function that returns 1 if argument is in the pipeline
		if ind == 0 && curNode.Func.Name == "exist" && len(curNode.Inputs) == 2 {
			if e != nil {
				curNode.Inputs[0] = curNode.Inputs[1]
			}
			e = nil
		}

		if e != nil {
			return e
		}
	}

	// check: are these operations: && || > >= = == != + - * / ^
	if curNode.Func != nil && utilities.Has(curNode.Func.Name, delim, operations) {
		return evalOps(curNode)
	}

	// is this a function eval?
	if curNode.Func != nil {
		return evalFunction(curNode)
	}

	if curNode.stet {
		return nil
	}

	// is it a constant?
	if evalConstant(curNode) {
		return nil
	}

	// must be a field from the pipeline then
	return fromPipeline(curNode, pipe)
}

// goNegative negates Value if Neg is true
func goNegative(x *Raw, neg bool) {
	if !neg {
		return
	}

	for ind := 0; ind < x.Len(); ind++ {
		switch v := x.Data[ind].(type) {
		case float64:
			x.Data[ind] = -v
		case float32:
			x.Data[ind] = -v
		case int32:
			x.Data[ind] = -v
		case int64:
			x.Data[ind] = -v
		}
		//		x.Data[ind] = -x.Data[ind].(float64)
	}
}

// matchedParen checks for mismatched parentheses
func matchedParen(expr string) error {
	if strings.Count(expr, "(") != strings.Count(expr, ")") {
		return fmt.Errorf("mismatched parentheses")
	}

	return nil
}

func one2Many(pipe Pipeline, rows int) (Pipeline, error) {
	if pipe.Rows() != 1 {
		return nil, fmt.Errorf("one2Many needs a pipe of Rows()=1")
	}

	gd := pipe.GData()
	dataNew := make([][]any, gd.FieldCount())
	for ind, fld := range gd.FieldList() {
		gdata, err := gd.GetRaw(fld)
		if err != nil {
			return nil, err
		}

		data := make([]any, rows)

		for indx := 0; indx < rows; indx++ {
			data[indx] = gdata.Data[0]
		}

		dataNew[ind] = data
	}

	newpipe, err := VecFromAny(dataNew, gd.FieldList(), pipe.GetFTypes())
	if err != nil {
		return nil, err
	}

	WithKeepRaw(true)(newpipe)

	return newpipe, nil
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
func AddToPipe(rootNode *OpNode, fieldName string, pipe Pipeline) (outPipe Pipeline, err error) {
	if rootNode.Raw == nil {
		return nil, fmt.Errorf("root node is nil")
	}

	if rootNode.Raw.Len() > 1 && pipe.Rows() > 1 && rootNode.Raw.Len() != pipe.Rows() {
		return nil, fmt.Errorf("AddtoPipe: exected length %d got length %d", pipe.Rows(), rootNode.Raw.Len())
	}

	if pipe.Rows() == 1 && rootNode.Raw.Len() > 1 {
		if pipe, err = one2Many(pipe, rootNode.Raw.Len()); err != nil {
			return nil, err
		}
	}

	// drop if already there
	_ = pipe.GData().Drop(fieldName)

	rawx := rootNode.Raw.Data
	if len(rawx) == 1 {
		tmp := rawx[0]
		rawx = make([]any, pipe.Rows())
		for ind := 0; ind < pipe.Rows(); ind++ {
			rawx[ind] = tmp
		}
	}

	role := rootNode.Role
	if rootNode.Role == FREither {
		switch rootNode.Raw.Data[0].(type) {
		case float64, float32, int, int32, int64:
			role = FRCts
		default:
			role = FRCat
		}
	}

	var fp *FParam
	normalize := false

	// see if in the pipeline
	if ft := pipe.GetFType(fieldName); ft != nil {
		fp = ft.FP
		normalize = ft.Normalized
	}

	// see in in pre-set FTypes

	if role == FRCat {
		err = pipe.GData().AppendD(NewRaw(rawx, nil), fieldName, fp, pipe.GetKeepRaw())
		return pipe, err
	}

	err = pipe.GData().AppendC(NewRaw(rawx, nil), fieldName, normalize, fp, pipe.GetKeepRaw())
	return pipe, err
}

// setValue sets the value of the loop variable
func setValue(loopVar string, val int, op *OpNode) {
	if op.Expression == loopVar {
		op.Raw = NewRaw([]any{val}, nil)
		op.stet = true // instructs Evaluate to keep this value
	}

	for ind := 0; ind < len(op.Inputs); ind++ {
		setValue(loopVar, val, op.Inputs[ind])
	}
}

// Loop enables looping in parse.  The ops in inner are run for each iteration.
//   - inner - is a slice of *OpNode expressions to run in the loop and then assign to "assign" in the pipeline
//   - loopVar - the name of the loop variable.  This may be used in the "inner" expressions. It is not added to the pipeline.
//   - loopVar takes on values from start to end.
func Loop(loopVar string, start, end int, inner []*OpNode, assign []string, pipe Pipeline) error {
	if inner == nil || assign == nil {
		return fmt.Errorf("assign and/or inner are nil")
	}

	if len(inner) != len(assign) {
		return fmt.Errorf("assign and inner must have the same length")
	}

	for loopInd := start; loopInd < end; loopInd++ {
		for nodeInd := 0; nodeInd < len(inner); nodeInd++ {
			var e error
			setValue(loopVar, loopInd, inner[nodeInd])

			if ex := Evaluate(inner[nodeInd], pipe); ex != nil {
				return ex
			}

			// if there, must drop it
			_ = pipe.GData().Drop(assign[nodeInd])

			if pipe, e = AddToPipe(inner[nodeInd], assign[nodeInd], pipe); e != nil {
				return e
			}
		}
	}

	return nil
}

// CopyNode copies an *OpNode tree (with no shared addresses)
func CopyNode(src *OpNode) (dest *OpNode) {
	dest = &OpNode{}
	dest.Expression = src.Expression
	dest.Neg = src.Neg
	dest.stet = src.stet
	dest.Role = src.Role

	if src.Func != nil {
		dest.Func = &FuncSpec{
			Name:   src.Func.Name,
			Return: src.Func.Return,
			Args:   nil,
			Level:  src.Func.Level,
		}
		dest.Func.Args = make([]reflect.Kind, len(src.Func.Args))
		copy(dest.Func.Args, src.Func.Args)
	}

	if src.Raw != nil {
		anySlice := make([]any, src.Raw.Len())
		copy(anySlice, src.Raw.Data)
		dest.Raw = NewRaw(anySlice, nil)
	}

	if src.Inputs == nil {
		return dest
	}

	dest.Inputs = make([]*OpNode, len(src.Inputs))

	for ind := 0; ind < len(src.Inputs); ind++ {
		dest.Inputs[ind] = CopyNode(src.Inputs[ind])
	}

	return dest
}

func newPlot() *Raw {
	ret := NewRaw([]any{1}, nil)

	fig = &grob.Fig{}

	return ret
}

func plotLine(y, lineType, color *Raw) (*Raw, error) {
	x := make([]any, y.Len())
	for ind := 0; ind < len(x); ind++ {
		x[ind] = float64(ind + 1)
	}

	xRaw := NewRaw(x, nil)
	return plotXY(xRaw, y, lineType, color)
}

func plotXY(x, y, lineType, color *Raw) (*Raw, error) {
	var sType grob.ScatterMode

	ret := NewRaw([]any{1}, nil)

	marker := strings.ToLower(utilities.Any2String(lineType.Data[0]))
	if !utilities.Has(marker, ",", mType) {
		return ret, fmt.Errorf("line type must be 'line' or 'markers', got %s", marker)
	}

	if x.Len() != y.Len() {
		return ret, fmt.Errorf("plotXY slices not same length: %d, %d", x.Len(), y.Len())
	}

	sColor := strings.ToLower(utilities.Any2String(color.Data[0]))
	if !utilities.Has(sColor, ",", colors) {
		return ret, fmt.Errorf("color %s not supported", sColor)
	}

	switch marker {
	case "markers":
		sType = grob.ScatterModeMarkers
	default:
		sType = grob.ScatterModeLines
	}

	tr := &grob.Scatter{
		Type: grob.TraceTypeScatter,
		X:    x.Data,
		Y:    y.Data,
		Name: "Scatter",
		Mode: sType,
		Line: &grob.ScatterLine{Color: sColor},
	}

	fig.AddTraces(tr)

	return ret, nil
}

func histogram(x, color, norm *Raw) (*Raw, error) {
	const normalized = "counts,percent,density"

	ret := NewRaw([]any{1}, nil)

	if x.Kind != reflect.Float64 {
		return ret, fmt.Errorf("histogram only for float64 currently, got %v", x.Kind)
	}

	sColor := strings.ToLower(utilities.Any2String(color.Data[0]))
	if !utilities.Has(sColor, ",", colors) {
		return ret, fmt.Errorf("color %s not supported", sColor)
	}

	sNorm := strings.ToLower(utilities.Any2String(norm.Data[0]))
	if !utilities.Has(sNorm, ",", normalized) {
		return ret, fmt.Errorf("unknown density normalization: %s", sNorm)
	}

	var normGrob grob.HistogramHistnorm
	switch sNorm {
	case "count":
		normGrob = grob.HistogramHistnormEmpty
	case "density":
		normGrob = grob.HistogramHistnormDensity
	case "percent":
		normGrob = grob.HistogramHistnormPercent
	}

	xRaw, err := utilities.AnySlice2Float64(x.Data)
	if err != nil {
		return ret, err
	}

	tr := &grob.Histogram{X: xRaw, Type: grob.TraceTypeHistogram,
		Histnorm: normGrob,
		Marker:   &grob.HistogramMarker{Color: sColor}}

	fig.AddTraces(tr)

	return ret, nil
}

func render(fileName, title, xlab, ylab *Raw) (*Raw, error) {
	ret := NewRaw([]any{1}, nil)

	sFile := utilities.Any2String(fileName.Data[0])
	sTitle := utilities.Any2String(title.Data[0])
	sXlab := utilities.Any2String(xlab.Data[0])
	sYlab := utilities.Any2String(ylab.Data[0])

	show := sFile == ""

	pd := &utilities.PlotDef{
		Show:     show,
		Title:    sTitle,
		XTitle:   sXlab,
		YTitle:   sYlab,
		STitle:   "",
		Legend:   false,
		Height:   Height,
		Width:    Width,
		FileName: sFile,
	}

	return ret, utilities.Plotter(fig, nil, pd)
}

func setPlotDim(width, height *Raw) (*Raw, error) {
	var (
		w, h *float64
		err  error
	)
	ret := NewRaw([]any{1}, nil)

	if w, err = utilities.Any2Float64(width.Data[0]); err != nil {
		return ret, fmt.Errorf("illegal float: %v", width.Data[0])
	}

	if h, err = utilities.Any2Float64(height.Data[0]); err != nil {
		return ret, fmt.Errorf("illegal float: %v", height.Data[0])
	}

	if *w <= 100 || *w >= 2000 {
		return ret, fmt.Errorf("plot width must be between 100 & 2000, got %v", w)
	}

	if *h <= 100 || *h >= 2000 {
		return ret, fmt.Errorf("plot height must be between 100 & 2000, got %v", w)
	}

	Height = *h
	Width = *w

	return ret, nil
}
