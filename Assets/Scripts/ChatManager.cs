#if !UNITY_WEBGL
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
#endif
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using TMPro;
using StarterAssets;
using System;
using UnityEngine.InputSystem;
using System.Net;
using System.Collections.Generic;
using Proyecto26;

public class ChatManager : MonoBehaviour
{
    public static bool RemoteCertificateValidationCallback(object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors) => true;

    public string apiUrl = "https://api.openai.com/v1/chat/completions";//"192.168.0.77:7412/v1/completions";
    public string apiKey;
    public InputField inputField;
    public TMP_Text chatLog;
    public ScrollRect scrollRect;
    private bool isTyping = false;
    
void Start()
{
    #if !UNITY_WEBGL
    ServicePointManager.ServerCertificateValidationCallback = ServerCertificateValidationCallback;
    ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls11 | SecurityProtocolType.Tls;
    #endif
}

    //public static bool RemoteCertificateValidationCallback(object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors) => true;


    public void ClickButton(){
        Debug.Log("cliiiiiiick");
    }

    public void SendChatMessage()
    {
        StartCoroutine(SendMessageCoroutine(inputField.text));
        inputField.text = "";
        inputField.DeactivateInputField();
        isTyping = false;
    }
    
    private void Update()
    {
    // Check if Shift + Space is pressed
     if (Keyboard.current.spaceKey.isPressed && Keyboard.current.tabKey.wasPressedThisFrame)
    {
        Debug.LogWarning("Keys pressed");
        // Toggle the isTyping state
        isTyping = !isTyping;
 
        // Enable or disable input based on the isTyping state
        if (isTyping)
        {
            // Enable typing in the input field
            inputField.Select();
            Debug.LogWarning("passed selection");
            inputField.ActivateInputField();
            
        }
        else
        {
            // Disable typing in the input field
            inputField.DeactivateInputField();
            
        }
    }
}
#if !UNITY_WEBGL
private bool ServerCertificateValidationCallback(object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors)
{
    // Return true to accept all certificates (insecure)
    return true;
}
#endif

    
public class AIResponse
{
    public List<AIChoice> choices;
    public string id;
    [UnityEngine.SerializeField, UnityEngine.Serialization.FormerlySerializedAs("object")]
    public string objectName;
    public int created;
    public string model;
    public Usage usage;
}



public class AIChoice
{
    public string text;
    public int index;
    public object logprobs;
    public string finish_reason;
}

public class Usage
{
    public int prompt_tokens;
    public int completion_tokens;
    public int total_tokens;
}



        

private IEnumerator SendMessageCoroutine(string message)
{
    // Display the user's message in the chat log
    chatLog.text += $"You: {message}\n";
    yield return null; // Ensure the chat log is updated before scrolling
    scrollRect.verticalNormalizedPosition = 0;

    string combinedPrompt = $"### Instructions:\n{chatLog.text.Trim()}\n### Response:\n";
    ChatMessage chatMessage = new ChatMessage
    {
        prompt = combinedPrompt,
        stop = new List<string> { "###" }
    };
    string json = JsonUtility.ToJson(chatMessage);
    Debug.Log("JSON Payload: " + json); // Log JSON payload

    // Use RestClient to send the POST request
    RestClient.Post(new RequestHelper
    {
        Uri = apiUrl,
        Method = "POST",
        ContentType = "application/json",
        BodyString = json,
        Headers = new Dictionary<string, string>
    {
        { "Authorization", apiKey }
    }
    })
    .Then(response =>
{
    Debug.Log("AI Response: " + response.Text);

    AIResponse aiResponse = JsonConvert.DeserializeObject<AIResponse>(response.Text);
    if (aiResponse.choices != null)
    {
        Debug.Log("Choices count: " + aiResponse.choices.Count); // Debug line to check the number of choices
        if (aiResponse.choices.Count > 0)
        {
            string aiText = aiResponse.choices[0].text.Trim();
            Debug.Log("Parsed AI Text: " + aiText); // Debug line to check the parsed AI text
            chatLog.text += $"AI: {aiText}\n";
            scrollRect.verticalNormalizedPosition = 0;
        }
        else
        {
            Debug.LogWarning("AI response is empty.");
        }
    }
    else
    {
        Debug.LogWarning("AI response couldn't be parsed.");
    }
})





    .Catch(error =>
    {
        Debug.LogError("Error: " + error.Message);
    });
}



}

public class ChatMessage
{
    public string prompt;
    public List<string> stop = new List<string> { "\n", "###" };
}

