package main
import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"github.com/openai/openai-go/openai"
)
var (
	useAIGateway    = flag.Bool("use-ai-gateway", false, "Use AI Gateway instead of direct Bedrock")
	aiGatewayURL    = flag.String("ai-gateway-url", "", "AI Gateway URL")
	awsAccessKeyID  = flag.String("aws-access-key-id", "", "AWS Access Key ID")
	awsSecretKey    = flag.String("aws-secret-key", "", "AWS Secret Key")
	awsSessionToken = flag.String("aws-session-token", "", "AWS Session Token (optional)")
	modelName       = flag.String("model-name", "eu.anthropic.claude-3-5-sonnet-20240620-v1:0", "Bedrock model name")
	toolURL         = flag.String("tool-url", "", "External tool URL for weather service")
)
const question = "What is the weather in New York City?"

func main() {
	flag.Parse()
	if *useAIGateway {
		log.Println("Using AI Gateway for requests.")
	} else {
		log.Println("Using Amazon Bedrock for requests.")
	}
	// Step 1: Send initial request
	response, err := sendRequest()
	if err != nil {
		log.Fatalf("Error sending request: %v", err)
	}
	// Step 2: Process tool call (if any)
	toolResponse := processToolCall(response)
	// Step 3: Send final request with tool response
	finalResponse, err := sendFinalRequest(toolResponse)
	if err != nil {
		log.Fatalf("Error sending final request: %v", err)
	}
	log.Println("Final Response from Model:", finalResponse)
}
// sendRequest sends the request either to AI Gateway or directly to Bedrock
func sendRequest() (map[string]interface{}, error) {
	if *useAIGateway {
		return sendAIGatewayRequest()
	}
	return sendBedrockRequest()
}
// sendAIGatewayRequest sends the request to AI Gateway
func sendAIGatewayRequest() (map[string]interface{}, error) {
	token := os.Getenv("TOKEN")
	// TODO: dont error on this if not using ai-gateway
	if token == "" {
		log.Fatal("TOKEN environment variable is required for AI Gateway.")
	}
	payload := map[string]interface{}{
		"model":   *modelName,
		"messages": []map[string]string{
			{"role": "user", "content": question},
		},
		"stream": false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/v1/chat/completions", *aiGatewayURL), bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var responseData map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseData)
	return responseData, err
}
// sendBedrockRequest sends the request directly to Amazon Bedrock
func sendBedrockRequest() (map[string]interface{}, error) {
	client := openai.NewClient(openai.ClientOptions{
		APIKey: *awsAccessKeyID,
	})
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
							"location": map[string]string{
								"type": "string",
							},
						},
						"required": []string{"location"},
					}),
				}),
			},
		}),
		Model: openai.F(*modelName),
	}
	resp, err := client.Chat.Create(context.Background(), params)
	if err != nil {
		return nil, err
	}
	return resp.Raw, nil
}
// processToolCall checks if the tool is called and gets weather data
func processToolCall(response map[string]interface{}) string {
	toolCall, exists := response["tool_calls"]
	if !exists {
		log.Println("No tool call detected. Using default response.")
		return "The weather is sunny, 25°C."
	}
	log.Println("Tool call detected. Fetching weather data...")
	if *toolURL != "" {
		return fetchWeatherFromService()
	}
	log.Println("Using mock weather response.")
	return "The weather in New York City is 22°C with scattered clouds."
}

// fetchWeatherFromService calls an external weather API
func fetchWeatherFromService() string {
	resp, err := http.Get(*toolURL)
	if err != nil {
		log.Println("Error fetching from tool URL:", err)
		return "Weather data unavailable."
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Println("Error reading response:", err)
		return "Weather data unavailable."
	}
	log.Println("Received weather data:", string(body))
	return string(body)
}
// sendFinalRequest sends the final response to the model
func sendFinalRequest(weather string) (string, error) {
	client := openai.NewClient(openai.ClientOptions{
		APIKey: *awsAccessKeyID,
	})
	params := openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(fmt.Sprintf("The weather tool responded: %s", weather)),
		}),
		Model: openai.F(*modelName),
	}
	resp, err := client.Chat.Create(context.Background(), params)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%v", resp.Raw), nil
}

//TODO: do I need the route rules? (if using ai-gateway)
func writeCredentials() error {
	if *awsAccessKeyID == nil || *awsSecretKey == nil {
		return fmt.Errorf("AWS Access Key ID and Secret Key must not be empty")
	}
	awsCredentialsBody := fmt.Sprintf("[default]\nAWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\n",
		*awsAccessKeyID, *awsSecretKey)
	if *awsSessionToken != "" {
		awsCredentialsBody += fmt.Sprintf("AWS_SESSION_TOKEN=%s\n", *awsSessionToken)
	}
	awsFilePath := os.TempDir() + "/aws-credential-file"
	awsFile, err := os.Create(awsFilePath)
	if err != nil {
		return err
	}
	defer awsFile.Close()
	_, err = awsFile.WriteString(awsCredentialsBody)
	if err != nil {
		return err
	}
	return nil
}


