package main
import (
	"context"
	"flag"
	"fmt"
	"log"
	"encoding/json"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)
var (
	useAIGateway    = flag.Bool("use-ai-gateway", true, "Use AI Gateway instead of direct Bedrock")
	aiGatewayURL    = flag.String("ai-gateway-url", "http://localhost:8080", "AI Gateway URL")
	awsAccessKeyID  = flag.String("aws-access-key-id", "", "AWS Access Key ID")
	awsSecretKey    = flag.String("aws-secret-key", "", "AWS Secret Key")
	awsSessionToken = flag.String("aws-session-token", "", "AWS Session Token (optional)")
	modelName       = flag.String("model-name", "eu.anthropic.claude-3-5-sonnet-20240620-v1:0", "Bedrock model name")
	toolURL         = flag.String("tool-url", "", "External tool URL for weather service")
)
const question = "What is the weather in New York City?"

func main() {
	flag.Parse()

	// Determine base URL (AI Gateway or Bedrock)
	baseURL := ""
	if *useAIGateway {
		log.Println("Using AI Gateway for requests.")
		baseURL = *aiGatewayURL + "/v1/"
	} else {
		log.Println("Using Amazon Bedrock for requests.")
	}

	// Initialize OpenAI client
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
	)

	params := openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		}),
		Tools: openai.F([]openai.ChatCompletionToolParam{
			{
				Type: openai.F(openai.ChatCompletionToolTypeFunction),
				Function: openai.F(openai.FunctionDefinitionParam{
					Name:        openai.String("get_weather"),
					Description: openai.String("Get weather at the given location"),
					Parameters: openai.F(openai.FunctionParameters{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]string{"type": "string"},
						},
						"required": []string{"location"},
					}),
				}),
			},
		}),
		Model: openai.F("eu.anthropic.claude-3-5-sonnet-20240620-v1:0"),
	}

	// Step 1: Send initial request
	response, err := sendRequest(client, params)
	if err != nil {
		log.Fatalf("Error sending request: %v", err)
	}

	fmt.Println(response.Choices[0].Message)

	toolCalls := response.Choices[0].Message.ToolCalls
	params.Messages.Value = append(params.Messages.Value, response.Choices[0].Message)
	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "get_weather" {
			// Extract the location from the function call arguments
			var args map[string]interface{}
			if argErr := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); argErr != nil {
				log.Println("Error unmarshalling the function arguments: %v", argErr)
			}
			location := args["location"].(string)
			if location != "New York City" {
				log.Println("Expected location to be New York City but got %s", location)
			}
			// Simulate getting weather data
			weatherData := "Sunny, 25°C"
			params.Messages.Value = append(params.Messages.Value, openai.ToolMessage(toolCall.ID, weatherData))
			log.Println("Appended tool message:", openai.ToolMessage(toolCall.ID, weatherData)) // Debug log
		}
	}

	// Step 3: Send final request with tool response
	finalResponse, err := sendFinalRequest(client, params)
	if err != nil {
		log.Fatalf("Error sending final request: %v", err)
	}
	log.Println("Final Response from Model:", finalResponse)
}

// sendRequest sends the request using OpenAI client
func sendRequest(client *openai.Client, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	resp, err := client.Chat.Completions.New(context.Background(), params)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// sendFinalRequest sends the tool response back to the model using OpenAI client
func sendFinalRequest(client *openai.Client, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	resp, err := client.Chat.Completions.New(context.Background(), params)
	if err != nil {
		return &openai.ChatCompletion{}, err
	}

	return resp, nil
}
