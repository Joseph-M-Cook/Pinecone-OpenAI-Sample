using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text;
using System.Text.Json;
using System.Net.Http.Headers;
using System.IO;

class Program
{
    // API Keys and Environments
    static string OPENAI_API_KEY = "";
    static string PINECONE_API_KEY = "";
    static string PROJECT_NAME = "";

    // Function to describe Pinecone index stats
    static async Task GetIndexInfo(){
        using (HttpClient client = new HttpClient())
        {
            try 
            {
                var request = new HttpRequestMessage
                {
                    Method = HttpMethod.Post,
                    RequestUri = new Uri($"https://{PROJECT_NAME}/describe_index_stats"),
                    Headers ={{ "accept", "application/json" },{ "Api-Key", PINECONE_API_KEY}}
                };
                using (var response = await client.SendAsync(request))
                {
                    response.EnsureSuccessStatusCode();
                    var body = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"Index Info: {body}");
                }
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine("Failed to get index info...");
                Console.WriteLine($"Request exception: {e.Message}");
            }
        }
    }

    // Function to clear all vectors from an index
    static async Task ClearIndex(){
        using (HttpClient client = new HttpClient())
        {
            try 
            {
                var request = new HttpRequestMessage
                {
                    Method = HttpMethod.Post,
                    RequestUri = new Uri($"https://{PROJECT_NAME}/vectors/delete"),
                    Headers ={{ "accept", "application/json" },{ "Api-Key", PINECONE_API_KEY}},
                    Content = new StringContent("{\"deleteAll\":true}"){Headers ={ContentType = new MediaTypeHeaderValue("application/json")}}
                };
                using (var response = await client.SendAsync(request))
                {
                    response.EnsureSuccessStatusCode();
                    var body = await response.Content.ReadAsStringAsync();
                    Console.WriteLine("Index Cleared Successfully");
                }
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine("Failed to clear index...");
                Console.WriteLine($"Request exception: {e.Message}");
            }
        }
    }
    
    // Function to get embedding from OpenAI Embeddings API
    static async Task<List<float>> GetEmbedding(string text){
        using (HttpClient client = new HttpClient())
        {
            try 
            {
                var request = new HttpRequestMessage
                {
                    Method = HttpMethod.Post,
                    RequestUri = new Uri("https://api.openai.com/v1/embeddings"),
                    Headers = { { "Authorization", $"Bearer {OPENAI_API_KEY}" },{ "accept", "application/json" }},
                    Content = new StringContent(JsonSerializer.Serialize(new { input = text, model = "text-embedding-ada-002" }), Encoding.UTF8, "application/json")
                };

                using (var response = await client.SendAsync(request))
                {
                    response.EnsureSuccessStatusCode();
                    var body = await response.Content.ReadAsStringAsync();
                    var jsonDocument = JsonDocument.Parse(body);
                    var vec = jsonDocument.RootElement.GetProperty("data")[0].GetProperty("embedding");
                    Console.WriteLine("Embedding Created Successfully");

                    var deserializedList = JsonSerializer.Deserialize<List<float>>(vec.ToString());
                    if (deserializedList != null)
                    {
                        return deserializedList;
                    }
                    else
                    {
                        Console.WriteLine("Deserialization failed. Returning an empty list.");
                        return new List<float>();
                    }
                }
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine("Failed to get embedding...");
                Console.WriteLine($"Request exception: {e.Message}");
                return new List<float>();
            }
        }
    }

    // Vector object
    public class Vector{
        public string Id { get; set; }
        public string Text { get; set; }
        public List<float> Values { get; set; }
        public Dictionary<string, int> Metadata { get; set; }

        public Vector(string id, string text, List<float> values, Dictionary<string, int> metadata)
        {
            Id = id;
            Text = text;
            Values = values;
            Metadata = metadata;
        }
    }

    // Function to split text from file into chunks
    public static List<string> SplitTextFileIntoChunks(string filePath){
        string text = File.ReadAllText(filePath);
        char[] delimiters = {'.'};
        string[] sentences = text.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
        
        for (int i = 0; i < sentences.Length; i++)
        {
            sentences[i] = sentences[i].Trim();
        }
        return new List<string>(sentences);
    }

    // Process chunks to vectors
    static async Task<List<Vector>> ChunksToVectors(List<string> chunks){
        List<Vector> vectors = new List<Vector>();
        List<List<float>> embeddings = await GetEmbeddings(chunks);
        
        for (int i = 0; i < chunks.Count; i++)
        {
            Console.WriteLine($"Chunk {i}: {chunks[i].Substring(0, Math.Min(chunks[i].Length, 50))}");
            List<float> Embedding = embeddings[i];
            string id = $"test_{i}";
            var metadata = new Dictionary<string, int> { { "Test", i } };
            Vector vector = new Vector(id, chunks[i], Embedding, metadata);
            vectors.Add(vector); 
        }
        return vectors;
    }

    // Function to get embedding from OpenAI Embeddings API
    static async Task<List<List<float>>> GetEmbeddings(List<string> stringsToEmbed){
        const int batchSize = 20;
        var allEmbeddings = new List<List<float>>();

            using (HttpClient client = new HttpClient())
            {
                for (int i = 0; i < stringsToEmbed.Count; i += batchSize)
                {
                    var batch = stringsToEmbed.Skip(i).Take(Math.Min(batchSize, stringsToEmbed.Count - i)).ToList();

                    try 
                    {
                        var request = new HttpRequestMessage
                        {
                            Method = HttpMethod.Post,
                            RequestUri = new Uri("https://api.openai.com/v1/embeddings"),
                            Headers = { { "Authorization", $"Bearer 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" },{ "accept", "application/json" }},
                            Content = new StringContent(JsonSerializer.Serialize(new { input = batch, model = "text-embedding-ada-002" }), Encoding.UTF8, "application/json") // Fix
                        };

                        using (var response = await client.SendAsync(request))
                        {
                            response.EnsureSuccessStatusCode();
                            var body = await response.Content.ReadAsStringAsync();
                            var jsonDocument = JsonDocument.Parse(body);

                            var batchEmbeddings = new List<List<float>>();
                            foreach (var data in jsonDocument.RootElement.GetProperty("data").EnumerateArray())
                            {
                                var vec = data.GetProperty("embedding");
                                var deserializedList = JsonSerializer.Deserialize<List<float>>(vec.ToString());
                                if (deserializedList != null)
                                {
                                    batchEmbeddings.Add(deserializedList);
                                }
                                else
                                {
                                    Console.WriteLine("Deserialization failed. Skipping the null reference.");
                                }
                            }

                            Console.WriteLine("Embedding Batch Complete");
                            allEmbeddings.AddRange(batchEmbeddings);
                        }
                    }
                    catch (HttpRequestException e)
                    {
                        Console.WriteLine("Failed to get embedding...");
                        Console.WriteLine($"Request exception: {e.Message}");
                    }
                }
            }
            return allEmbeddings;
    }

    // Function to upsert a list of vectors in batches
    static async Task UpsertVectors(List<Vector> vectors){
        const int batchSize = 100;

        using (HttpClient client = new HttpClient())
        {
            try
            {
                for (int i = 0; i < vectors.Count; i += batchSize)
                {
                    List<Vector> batch = vectors.Skip(i).Take(batchSize).ToList();

                    var request = new HttpRequestMessage
                    {
                        Method = HttpMethod.Post,
                        RequestUri = new Uri($"https://{PROJECT_NAME}/vectors/upsert"),
                        Headers ={{ "accept", "application/json" },{ "Api-Key", PINECONE_API_KEY }},
                        Content = new StringContent(JsonSerializer.Serialize(new { vectors = batch.Select(v => new { id = v.Id, values = v.Values, metadata = v.Metadata }) }))
                        {
                            Headers = { ContentType = new MediaTypeHeaderValue("application/json") }
                        }
                    };

                    using (var response = await client.SendAsync(request))
                    {
                        response.EnsureSuccessStatusCode();
                        var body = await response.Content.ReadAsStringAsync();
                        Console.WriteLine(body);
                    }
                    Console.WriteLine("\nBatch Complete\n");
                }
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine($"Request exception: {e.Message}");
            }
        }
    }
    
    // Function to perform Pinecone vector DB query and return list of IDs
    static async Task<List<string>> PineconeQuery(Vector vector){
        List<string> ids = new List<string>();

        using (HttpClient client = new HttpClient())
        {
            try
            {
                var request = new HttpRequestMessage
                {
                    Method = HttpMethod.Post,
                    RequestUri = new Uri($"https://{PROJECT_NAME}/query"),
                    Headers ={{ "accept", "application/json" },{ "Api-Key", PINECONE_API_KEY }},
                    Content = new StringContent(JsonSerializer.Serialize(new
                    {
                        vector = vector.Values,
                        includeMetadata = true,
                        topK = 5
                    }))
                    {
                        Headers =
                        {
                            ContentType = new MediaTypeHeaderValue("application/json")
                        }
                    }
                };

                using (var response = await client.SendAsync(request))
                {
                    response.EnsureSuccessStatusCode();
                    var body = await response.Content.ReadAsStringAsync();
                    Console.WriteLine("Query Embedded Successfully");
                    Console.WriteLine();

                    var result = JsonSerializer.Deserialize<JsonDocument>(body);
                    if (result != null)
                    {
                        var matches = result.RootElement.GetProperty("matches");

                        foreach (var match in matches.EnumerateArray())
                        {
                            var id = match.GetProperty("id").GetString();
                            if (id != null)
                            {
                                ids.Add(id);
                            }
                        }
                    }

                }
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine("Failed to perform query...");
                Console.WriteLine($"Request exception: {e.Message}");
            }
        }

        return ids;
    }

    // Function that takes IDs and finds associated text
    static List<string> FindMatchingText(List<string> IDs, List<Vector> vectors){
        List<string> matchingTexts = new List<string>();

        foreach (var id in IDs)
        {
            Vector? vector = vectors.FirstOrDefault(v => v.Id == id); 

            if (vector != null)
            {
                string? text = vector.Text; 

                if (text != null)
                {
                    matchingTexts.Add(text);
                }
            }
        }

        return matchingTexts;
    }

    // Function to find context for a query
    static async Task<string> ProcessQuery(string user_query, List<Vector> vectors){
        List<float> queryValues = await GetEmbedding(user_query);
        Vector queryVector = new Vector("Query", user_query, queryValues, new Dictionary<string, int> { { "Query", -1} });
        List<string> matches = await PineconeQuery(queryVector);
        List<string> matchingTexts = FindMatchingText(matches, vectors);
        string prompt = GeneratePrompt(matchingTexts, user_query);
        Console.WriteLine(prompt);
        string response = CallGPT(prompt);
        string message = ProcessMessage(response);

        Console.WriteLine("Response Message: " + message);
        CostEvaluation(response);

        return message;
    }

    // Function to generate prompt given a list of context and query
    static string GeneratePrompt(List<string> contexts, string userQuery){
        StringBuilder promptBuilder = new StringBuilder();

        for (int i = 0; i < contexts.Count; i++)
        {
            string context = contexts[i];
            promptBuilder.AppendLine($"Context: {context}");
        }

        promptBuilder.AppendLine();
        promptBuilder.AppendLine($"Question: {userQuery}");
        promptBuilder.AppendLine();
        promptBuilder.AppendLine("Please provide a detailed and clear response to this question based on the given contexts.");

        return promptBuilder.ToString();
    }

    // Function to call ChatGPT API
    static string CallGPT(string prompt){
        // OpenAI API key
        using var httpClient = new HttpClient { BaseAddress = new Uri("https://api.openai.com/v1/") };
        httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", OPENAI_API_KEY);
        
        StringBuilder systemPromptBuilder = new StringBuilder();

        systemPromptBuilder.AppendLine("You're tasked with answering a user query.");
        systemPromptBuilder.AppendLine("You've been given some context obtained through a semantic search of text chunks from a vector database. ");
        systemPromptBuilder.AppendLine("Please consider all the provided contexts carefully and generate a comprehensive and accurate response.");

        // Set up the request params
        var requestBody = new{model = "gpt-3.5-turbo",messages = new[]
            {
                new { role = "system", content = systemPromptBuilder.ToString() },
                new { role = "user", content = prompt }
            }};

        var json = JsonSerializer.Serialize(requestBody);
        var response = httpClient.PostAsync("chat/completions", new StringContent(json, Encoding.UTF8, "application/json")).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();
        
        // Return JSON of the response
        return response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
    }

     // Function to process message
    static string ProcessMessage(string response){
        // Parse response from OpenAI API
        var jsonDocument = JsonDocument.Parse(response);

        // Extract the response
        var choices = jsonDocument.RootElement.GetProperty("choices");
        if (choices.GetArrayLength() > 0)
        {
            var firstChoice = choices[0];
            var messageProperty = firstChoice.GetProperty("message");

            if (!messageProperty.ValueKind.Equals(JsonValueKind.Null))
            {
                var contentProperty = messageProperty.GetProperty("content");
                if (!contentProperty.ValueKind.Equals(JsonValueKind.Null))
                {
                    var message = contentProperty.GetString();
                    if (message != null)
                    {
                        return message;
                    }
                }
            }
        }

        return string.Empty; // Default value if any property is null
    }

    // Cost evaluation
    static void CostEvaluation(string response){
        var jsonDocument = JsonDocument.Parse(response);

        // Fetch token amounts
        var promptTokens = jsonDocument.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();
        var completionTokens = jsonDocument.RootElement.GetProperty("usage").GetProperty("completion_tokens").GetInt32();

        // Display token amounts
        Console.WriteLine($"\n{"Prompt Tokens Used:",-35}{promptTokens:N0}");
        Console.WriteLine($"{"Completion Tokens Used:",-35}{completionTokens:N0}");
        Console.WriteLine($"{"Total Tokens Used:",-35}{promptTokens + completionTokens:N0}");

        // Calculate cost of GPT-3.5-turbo
        decimal gpt35TurboUsageCost = (promptTokens + completionTokens) * 0.0015m / 1000m;
        Console.WriteLine($"{"GPT-3.5-Turbo Usage Cost: ",-35}${gpt35TurboUsageCost:F6}");

        // Calculate cost of GPT-4
        decimal gpt4UsageCost = (promptTokens * 0.03m / 1000m) + (completionTokens * 0.06m / 1000m);
        Console.WriteLine($"{"GPT-4 Usage Cost: ",-35}${gpt4UsageCost:F6}\n");
    }

    // Main
    static async Task Main(string[] args){

        await GetIndexInfo();
        await ClearIndex();

        string filePath = "testingtext.txt";
        
        List<string> chunks = SplitTextFileIntoChunks(filePath);
        List<Vector> vectors = await ChunksToVectors(chunks);
        
        await UpsertVectors(vectors);

        string user_query = "Tell me about Alex's report on the building.";
        await ProcessQuery(user_query, vectors);

        user_query = "How can Lumina Towers reduce its carbon footprint and promote sustainability?";
        await ProcessQuery(user_query, vectors);

        user_query = "What measures can be taken to improve the energy efficiency of the building's lighting system?";
        await ProcessQuery(user_query, vectors);
        
    }
}
