package com.ollama.ollama.utils;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import static com.ollama.ollama.component.ApplicationProperties.AppName;
import static java.lang.ProcessBuilder.startPipeline;
import java.io.IOException;

public class OllamaReader {
    public String prompt;
    public String response;

    public OllamaReader(String prompt) throws IOException {
        String path = new File(".").getAbsolutePath();

        String appNameDir = "\\" + AppName + "\\";
        String psScript = AppName + ".ps1";

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
            int positionStart = text.indexOf("\"response\":", 0);

            if (positionStart != -1) {
                int positionEnd = text.indexOf(",\"done\":", positionStart) + ": ".length();

                if (positionEnd != -1) {
                    //response = text.substring(position);
                    response = text.substring(
                                positionStart + "\"response\":".length(),
                                positionEnd - ",".length()
                            ); //"Hello",
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
}
