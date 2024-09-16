package converter

import (
	"github.com/asynkron/protoactor-go/actor"
	"main.go/messages"
	"main.go/model"
)

func ConvertToProtoWeights(w model.Weights) *messages.Weights {

	var protoConv1Weights []*messages.ConvWeight
	for _, convLayer1 := range w.Conv1Weight {
		var protoKernels []*messages.Kernel
		for _, convLayer2 := range convLayer1 {
			var protoRows []*messages.Row
			for _, convLayer3 := range convLayer2 {
				var protoValues []float64
				for _, value := range convLayer3 {
					protoValues = append(protoValues, float64(value))
				}
				protoRows = append(protoRows, &messages.Row{Value: protoValues})
			}
			protoKernels = append(protoKernels, &messages.Kernel{Row: protoRows})
		}
		protoConv1Weights = append(protoConv1Weights, &messages.ConvWeight{Kernel: protoKernels})
	}

	var protoConv2Weights []*messages.ConvWeight
	for _, convLayer1 := range w.Conv2Weight {
		var protoKernels []*messages.Kernel
		for _, convLayer2 := range convLayer1 {
			var protoRows []*messages.Row
			for _, convLayer3 := range convLayer2 {
				var protoValues []float64
				for _, value := range convLayer3 {
					protoValues = append(protoValues, float64(value))
				}
				protoRows = append(protoRows, &messages.Row{Value: protoValues})
			}
			protoKernels = append(protoKernels, &messages.Kernel{Row: protoRows})
		}
		protoConv2Weights = append(protoConv2Weights, &messages.ConvWeight{Kernel: protoKernels})
	}

	convertFcWeights := func(fcWeights [][]float64) []*messages.FcWeight {
		var protoFcWeights []*messages.FcWeight
		for _, fcLayer1 := range fcWeights {
			var protoFcLayer1 []*messages.FcWeight2
			for _, fcValue := range fcLayer1 {
				protoFcLayer1 = append(protoFcLayer1, &messages.FcWeight2{Fc1Weight: float64(fcValue)})
			}
			protoFcWeights = append(protoFcWeights, &messages.FcWeight{Fc1Weight: protoFcLayer1})
		}
		return protoFcWeights
	}

	return &messages.Weights{
		Conv1Weight: protoConv1Weights,
		Conv1Bias:   w.Conv1Bias,
		Conv2Weight: protoConv2Weights,
		Conv2Bias:   w.Conv2Bias,
		Fc1Weight:   convertFcWeights(w.Fc1Weight),
		Fc1Bias:     w.Fc1Bias,
		Fc2Weight:   convertFcWeights(w.Fc2Weight),
		Fc2Bias:     w.Fc2Bias,
	}
}

func ConvertFromProtoWeights(pw *messages.Weights) model.Weights {
	var conv1Weights [][][][]float64
	for _, protoLayer1 := range pw.Conv1Weight {
		var convLayer1 [][][]float64
		for _, protoKernel := range protoLayer1.Kernel {
			var convLayer2 [][]float64
			for _, protoRow := range protoKernel.Row {
				convLayer2 = append(convLayer2, protoRow.Value)
			}
			convLayer1 = append(convLayer1, convLayer2)
		}
		conv1Weights = append(conv1Weights, convLayer1)
	}

	var conv2Weights [][][][]float64
	for _, protoLayer1 := range pw.Conv2Weight {
		var convLayer1 [][][]float64
		for _, protoKernel := range protoLayer1.Kernel {
			var convLayer2 [][]float64
			for _, protoRow := range protoKernel.Row {
				convLayer2 = append(convLayer2, protoRow.Value)
			}
			convLayer1 = append(convLayer1, convLayer2)
		}
		conv2Weights = append(conv2Weights, convLayer1)
	}

	// Convert Fc1Weight, Fc2Weight
	convertProtoFcWeights := func(protoFcWeights []*messages.FcWeight) [][]float64 {
		var fcWeights [][]float64
		for _, protoLayer1 := range protoFcWeights {
			var fcLayer1 []float64
			for _, protoFcWeight := range protoLayer1.Fc1Weight {
				fcLayer1 = append(fcLayer1, protoFcWeight.Fc1Weight)
			}
			fcWeights = append(fcWeights, fcLayer1)
		}
		return fcWeights
	}

	return model.Weights{
		Conv1Weight: conv1Weights,
		Conv1Bias:   pw.Conv1Bias,
		Conv2Weight: conv2Weights,
		Conv2Bias:   pw.Conv2Bias,
		Fc1Weight:   convertProtoFcWeights(pw.Fc1Weight),
		Fc1Bias:     pw.Fc1Bias,
		Fc2Weight:   convertProtoFcWeights(pw.Fc2Weight),
		Fc2Bias:     pw.Fc2Bias,
	}
}

func ConvertProtoToActorPID(protoPID *messages.PID) *actor.PID {
	return &actor.PID{
		Address:   protoPID.Address,
		Id:        protoPID.Id,
		RequestId: protoPID.RequestId,
	}
}

func ConvertActorToProtoPID(actorPID *actor.PID) *messages.PID {
	return &messages.PID{
		Address:   actorPID.Address,
		Id:        actorPID.Id,
		RequestId: actorPID.RequestId,
	}
}
