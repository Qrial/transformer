package main

import (
	"fmt"
	"math"
)

// 定义一个简化的张量类型，用于存储数据和执行基本运算
type Tensor []float64

// 矩阵乘法
func (a Tensor) MatMul(b Tensor) Tensor {
	rowsA := len(a) / len(b) // 假设 a 是 b 的行数的整数倍
	colsA := len(b)
	result := make(Tensor, rowsA*colsA)
	
    for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			sum := 0.0
			
            for k := 0; k < len(b); k++ {
				sum += float64(a[i*len(b)+k]) * float64(b[k*colsA+j])
			}
			
            result[i*colsA+j] = sum
		}
	}

	return result
}

// 缩放点积注意力机制
func ScaledDotProductAttention(q, k, v Tensor, scale float64) Tensor {
	scores := make(Tensor, len(q))
	
    for i := 0; i < len(q); i++ {
		sum := 0.0
		for j := 0; j < len(k); j++ {
			sum += float64(q[i]) * float64(k[j]) / scale
		}
		scores[i] = sum
	}

	// 应用 softmax 得到注意力权重
	weights := Softmax(scores)
	// 计算加权的值
	result := make(Tensor, len(v))
	
    for i := 0; i < len(v); i++ {
		sum := 0.0
		
        for j := 0; j < len(weights); j++ {
			sum += float64(weights[j]) * float64(v[j])
		}
		
        result[i] = sum
	}
	
    return result
}

// Softmax 函数
func Softmax(x Tensor) Tensor {
	maxVal := float64(0)
	
    for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	
    expSum := float64(0)
	result := make(Tensor, len(x))
	
    for i, v := range x {
		expVal := math.Exp(float64(v-maxVal))
		result[i] = expVal
		expSum += expVal
	}
	
    for i, v := range result {
		result[i] = v / expSum
	}
	
    return result
}

func main() {
	// 示例：简化的自注意力机制
	q := Tensor{1, 2, 3}
	k := Tensor{4, 5, 6}
	v := Tensor{7, 8, 9}

	scale := math.Sqrt(float64(len(k)))

	attn := ScaledDotProductAttention(q, k, v, scale)
	fmt.Println("Attention result:", attn)
}
