// Copyright 2019 Dmitry A. Mottl. All rights reserved.
// Use of this source code is governed by MIT license
// that can be found in the LICENSE file.

// Linear regression for 1D variable
package linreg

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// LinReg returns alpha (intersection), beta (slope) and variance for 1-D linear regression.
// Example:
//	func main() {
//		rand.Seed(time.Now().UTC().UnixNano())
//		var N int = 10
//		var X []float64 = make([]float64, N)
//		var Y []float64 = make([]float64, N)
//		const beta = 2
//		const alpha = 5
//		const std = beta / 5.0
//		for i:=0; i<N; i++ {
//			X[i] = float64(i)
//			Y[i] = beta * X[i] + alpha + rand.NormFloat64() * std
//		}
//
//		a_, b_, v := LinReg(X, Y)
//		mse := math.Sqrt(v)
//		fmt.Printf("a=%.3f, b=%.3f, mse=%.3f, std=%.3f\n", a_, b_, mse, float64(std))
//	}
func LinReg(x []float64, y []float64) (alpha float64, beta float64, variance float64) {
	var xy, x_, y_, x2_ float64
	if len(x) != len(y) {
		panic("The lengths of x and y need to be equal")
	}

	n := float64(len(x))

	for i, _ := range x {
		xy += x[i] * y[i] / n
		x2_ += x[i] * x[i] / n
		x_ += x[i]
		y_ += y[i]
	}
	x_ = x_ / n
	y_ = y_ / n

	den := x2_ - x_*x_

	beta = (xy - x_*y_) / den
	alpha = (y_*x2_ - x_*xy) / den

	for i, _ := range x {
		e := y[i] - x[i]*beta - alpha
		variance += e * e / n
	}
	return alpha, beta, variance
}
