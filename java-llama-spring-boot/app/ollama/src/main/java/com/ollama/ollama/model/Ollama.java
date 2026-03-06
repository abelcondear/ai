package com.ollama.ollama.model;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import static com.ollama.ollama.component.ApplicationProperties.AppName;
import static java.lang.ProcessBuilder.startPipeline;
import java.io.IOException;

public class Ollama {
    private final String prompt;
    private final String response;

    public Ollama(String prompt) throws IOException {
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
                                    //this.encodeUTF_8(String.format("\"%s\"", prompt))
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

                if (position != -1) { response = text.substring(position); }
            }
        }

        this.prompt = prompt;
        this.response = response;
    }

    private String encodeUTF_8(String text) {
        ByteBuffer buffer_ISD_8859_1 = StandardCharsets.ISO_8859_1.encode(text);
        String string_ISD_8859_1 = StandardCharsets.ISO_8859_1.decode(buffer_ISD_8859_1).toString();

        ByteBuffer buffer_UTF_8 = StandardCharsets.UTF_8.encode(string_ISD_8859_1);
        return StandardCharsets.UTF_8.decode(buffer_UTF_8).toString();
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
