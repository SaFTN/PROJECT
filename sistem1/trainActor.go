package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"sync"

	"github.com/asynkron/protoactor-go/actor"
	"main.go/converter"
	"main.go/messages"
	"main.go/model"
)

type DataSet struct {
	Images [][]float32
	Labels []int
}

type DataResponse struct {
	Images [][]float32 `json:"images"`
	Labels []int       `json:"labels"`
}

type TrainerActor struct {
	commActorPID *actor.PID
}

var dataset *DataSet = nil
var modelWeights *model.Weights = nil
var actorContext actor.Context = nil
var isServerActive bool = false
var trainerState *TrainerActor = nil

func (actor *TrainerActor) Receive(ctx actor.Context) {
	actorContext = ctx
	trainerState = actor
	switch msg := ctx.Message().(type) {
	case DataSet:
		fmt.Println("Trainer Actor received dataset")
		dataset = &msg
		fmt.Println("Trainer Actor starting with random weights")
		var wg sync.WaitGroup

		wg.Add(1)

		go initializeServer()
		go executePythonScript(&wg)

		wg.Wait()
		//ctx.Send(actor.commActorPID, msg)
	case model.LocalWeights:
		fmt.Println("Trainer Actor received weights")
		modelWeights = &msg.Weights
		var wg sync.WaitGroup

		wg.Add(1)

		go initializeServer()
		go executePythonScript(&wg)

		wg.Wait()
		fmt.Println("Python script has finished executing.")
	}
}

func executePythonScript(wg *sync.WaitGroup) {
	defer wg.Done()
	cmd := exec.Command("python", "train.py")

	cmd.Stdout = log.Writer()
	cmd.Stderr = log.Writer()

	err := cmd.Run()
	if err != nil {
		log.Fatalf("cmd.Run() failed with %s\n", err)
	}
}

func handleDataRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if dataset == nil {
		http.Error(w, "No dataset available", http.StatusNotFound)
		return
	}

	response := DataResponse{
		Images: dataset.Images,
		Labels: dataset.Labels,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, fmt.Sprintf("Error encoding response: %v", err), http.StatusInternalServerError)
		return
	}

	fmt.Println("Sending dataset for training")
}

func handleInitialWeightsRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if modelWeights == nil {
		http.Error(w, "No weights available", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(modelWeights); err != nil {
		http.Error(w, fmt.Sprintf("Error encoding weights: %v", err), http.StatusInternalServerError)
		return
	}
}

func handleWeightsUpdateRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var incomingWeights model.Weights
	if err := json.NewDecoder(r.Body).Decode(&incomingWeights); err != nil {
		http.Error(w, fmt.Sprintf("Error decoding weights: %v", err), http.StatusBadRequest)
		return
	}

	fmt.Println("Weights received successfully")
	modelWeights = &incomingWeights

	var localWeights model.LocalWeights
	localWeights.Weights = *modelWeights
	localWeightsMessage := &messages.LocalWeights{
		Weights: converter.ConvertToProtoWeights(localWeights.Weights),
	}
	actorContext.Send(trainerState.commActorPID, localWeightsMessage)

	w.WriteHeader(http.StatusOK)
}

func initializeServer() {
	if isServerActive {
		return
	}

	isServerActive = true

	http.HandleFunc("/mnist_data", handleDataRequest)
	http.HandleFunc("/weights", handleWeightsUpdateRequest)
	http.HandleFunc("/initial_weights", handleInitialWeightsRequest)

	port := ":8080"
	fmt.Printf("Server listening on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatal(err)
	}

	isServerActive = false
}
