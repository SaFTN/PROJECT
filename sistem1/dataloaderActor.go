package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/asynkron/protoactor-go/actor"
	"main.go/model"
)

var (
	imageFilePath = "data/train-images-idx3-ubyte"
	labelFilePath = "data/train-labels-idx1-ubyte"
)

var startIndex int = 0
var endIndex int = 20000

type LoadDataRequest struct{}

type DataLoaderActor struct {
	commActorPID *actor.PID
	trainingPID  *actor.PID
}

func (actor *DataLoaderActor) Receive(ctx actor.Context) {
	switch msg := ctx.Message().(type) {
	case LoadDataRequest:
		images, err := readIDXFile(imageFilePath)
		if err != nil {
			fmt.Printf("Error reading images file: %v\n", err)
			return
		}

		labels, err := readIDXFile(labelFilePath)
		if err != nil {
			fmt.Printf("Error reading labels file: %v\n", err)
			return
		}

		numRows, numCols := 28, 28
		imageSize := numRows * numCols

		dataSet := DataSet{
			Images: make([][]float32, endIndex-startIndex),
			Labels: make([]int, endIndex-startIndex),
		}

		for i := 0; i < endIndex-startIndex; i++ {
			dataSet.Images[i] = make([]float32, imageSize)
			for j := 0; j < imageSize; j++ {
				dataSet.Images[i][j] = float32(images[i*imageSize+j]) / 255.0
			}
			dataSet.Labels[i] = int(labels[i])
		}
		fmt.Printf("Loaded and sending dataset to training actor PID: %s\n", actor.trainingPID.Id)
		ctx.Send(actor.trainingPID, dataSet)
	case model.TrainingPIDMsg:
		actor.trainingPID = msg.TrainingPID
	}
}

func readIDXFile(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magicNumber int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("could not read magic number: %v", err)
	}

	var numItems int32
	if err := binary.Read(file, binary.BigEndian, &numItems); err != nil {
		return nil, fmt.Errorf("could not read number of items: %v", err)
	}

	switch magicNumber {
	case 2049:
		labels := make([]byte, numItems)
		if _, err := file.Read(labels); err != nil {
			return nil, fmt.Errorf("could not read labels data: %v", err)
		}
		return labels, nil
	case 2051:
		var numRows, numCols int32
		if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
			return nil, fmt.Errorf("could not read number of rows: %v", err)
		}
		if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
			return nil, fmt.Errorf("could not read number of columns: %v", err)
		}
		imageSize := numRows * numCols
		images := make([]byte, numItems*imageSize)
		if _, err := file.Read(images); err != nil {
			return nil, fmt.Errorf("could not read images data: %v", err)
		}
		return images, nil
	default:
		return nil, fmt.Errorf("unknown magic number %d", magicNumber)
	}
}
