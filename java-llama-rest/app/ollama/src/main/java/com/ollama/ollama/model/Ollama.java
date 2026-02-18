package com.ollama.ollama.model;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import com.ollama.ollama.component.ApplicationProperties;
import static java.lang.ProcessBuilder.startPipeline;
import java.io.IOException;

public class Ollama {
    private final String prompt;
    private final String response;

    public Ollama(String prompt) throws IOException {
        String path = new File(".").getAbsolutePath();

        String appNameDir = "\\" + ApplicationProperties.AppName + "\\";
        String psScript = ApplicationProperties.AppName + ".ps1";

        int index = path.indexOf(appNameDir);

        String currentPath = path.substring(0, index + appNameDir.length());
        String filePath = currentPath + psScript;

        List<ProcessBuilder> builders = Arrays.asList(
                    new ProcessBuilder(
                        "powershell",
                                    "-File",
                                    filePath,
                                    "-Prompt",
                                    String.format("\"%s\"", prompt)
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

        this.prompt = prompt;
        this.response = response;
    }

    private List<String> readOutput(InputStream inputStream) {
        Reader isr = new InputStreamReader(inputStream);
        BufferedReader r = new BufferedReader(isr);
        return r.lines().toList();
    }

    public String getResponse() {
        return this.response;
    }

    public String getPrompt() {
        return this.prompt;
    }
}
