package org.example;

import io.github.ollama4j.Ollama;
import io.github.ollama4j.models.chat.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {

        Ollama ollama = new Ollama("http://localhost:11434");
        ollama.setRequestTimeoutSeconds(120);

        String userPrompt = "Hello";

        System.out.println();
        System.out.println();

        System.out.print("User Prompt: " + userPrompt);
        System.out.println();

        List<OllamaChatMessage> messages = Arrays.asList(
                new OllamaChatMessage(OllamaChatMessageRole.SYSTEM, "You are an assistant."),
                new OllamaChatMessage(OllamaChatMessageRole.USER, userPrompt)
        );

        // Create the chat request object
        OllamaChatRequest chatRequest = new OllamaChatRequest();
        chatRequest.setMessages(messages);

        chatRequest.setModel("llama3");
        chatRequest.setStream(true);
        chatRequest.build();

        try {
            System.out.print("Assistant Response: ");

            OllamaChatTokenHandler tokenHandler;
            tokenHandler = new OllamaChatTokenHandler() {
                @Override
                public void accept(OllamaChatResponseModel ollamaChatResponseModel) {
                    System.out.print(ollamaChatResponseModel.getMessage().getResponse());
                }
            };

            // Send the request and get the result
            OllamaChatResult result = ollama.chat(chatRequest, tokenHandler);
        } catch (Exception e) {
            //e.printStackTrace();
            //System.out.println(e.getMessage());
            //TODO
        }

        System.out.println();
    }
}