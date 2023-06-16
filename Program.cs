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
    static async Task<List<float>> GetEmbedding(string text)
    {
        using (HttpClient client = new HttpClient())
        {
            try
            {
                var request = new HttpRequestMessage
                {
                    Method = HttpMethod.Post,
                    RequestUri = new Uri("https://api.openai.com/v1/embeddings"),
                    Headers = { { "Authorization", $"Bearer {OPENAI_API_KEY}" }, { "accept", "application/json" } },
                    Content = new StringContent(JsonSerializer.Serialize(new { input = text, model = "text-embedding-ada-002" }), Encoding.UTF8, "application/json")
                };

                using (var response = await client.SendAsync(request))
                {
                    response.EnsureSuccessStatusCode();
                    var body = await response.Content.ReadAsStringAsync();
                    var jsonDocument = JsonDocument.Parse(body);
                    var vec = jsonDocument.RootElement.GetProperty("data")[0].GetProperty("embedding");
                    Console.WriteLine("Embedding Created Successfully");
                    return JsonSerializer.Deserialize<List<float>>(vec.ToString()) ?? new List<float>();
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
        char[] delimiters = { '.' , '?', '!'};
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

        for (int i = 0; i < chunks.Count; i++)
        {
            Console.WriteLine($"Chunk {i}: {chunks[i].Substring(0, Math.Min(chunks[i].Length, 50))}");
            List<float> testEmbedding = await GetEmbedding(chunks[i]);
            string id = $"test_{i}";
            var metadata = new Dictionary<string, int> { { "Test", i } };
            Vector vector = new Vector(id, chunks[i], testEmbedding, metadata);
            vectors.Add(vector); 
        }
        return vectors;
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
    static async Task<List<string>> ProcessQuery(string user_query, List<Vector> vectors){
        List<float> queryValues = await GetEmbedding(user_query);
        Vector queryVector = new Vector("Query", user_query, queryValues, new Dictionary<string, int> { { "Query", -1 } } );
        List<string> matches = await PineconeQuery(queryVector);
        List<string> matchingTexts = FindMatchingText(matches, vectors);
        return matchingTexts;
    }


    // Main
    static async Task Main(string[] args){
        await GetIndexInfo();

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
