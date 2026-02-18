package com.ollama.ollama.model;

import java.io.*;
import java.util.Arrays;
import java.util.List;

import static java.lang.ProcessBuilder.startPipeline;

public class Ollama {
    private final String content;

    public Ollama(String content) throws IOException {
        List<ProcessBuilder> builders = Arrays.asList(
                    new ProcessBuilder(
                        "powershell",
                                    "-File",
                                    "C:\\Users\\scott\\Documents\\Me\\Programs\\backend\\java\\chat\\llama-chat" +
                                        ".v04\\ollama\\ollama.ps1",
                                    "-Prompt",
                                    String.format("\"%s\"", content)
                                    )
                  );

        List<Process> processes = startPipeline(builders);
        Process last = processes.getLast();

        List<String> output = readOutput(last.getInputStream());

        String response = "There is no response.";

        for (String text : output) {
            int position = text.indexOf("response", 0);

            if (position != -1) {
                position = text.indexOf(": ", position) + ": ".length();

                if (position != -1) {
                    response = text.substring(position);
                }
            }
        }

        this.content = response;
    }

    private List<String> readOutput(InputStream inputStream) {
        Reader isr = new InputStreamReader(inputStream);
        BufferedReader r = new BufferedReader(isr);
        return r.lines().toList();
    }

    public String getContent() {
        return content;
    }
}
