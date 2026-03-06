package com.ollama.ollama.utils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static com.ollama.ollama.component.ApplicationProperties.AppName;
import static java.lang.ProcessBuilder.startPipeline;
import org.apache.commons.lang3.StringUtils;
import java.io.IOException;

public class OllamaReader {
    public String prompt;
    public List<String> response;

    public OllamaReader(String prompt) throws IOException {
        String path = new File(".").getAbsolutePath();

        String appNameDir = "\\" + AppName + "\\";

        String psScript = AppName + ".ps1";
        String psTemplateScript = AppName + ".template.ps1";

        int index = path.indexOf(appNameDir);

        String currentPath = path.substring(0, index + appNameDir.length());
        String filePath = currentPath + psScript;

        // restore psScript from template script
        Files.copy(
                Paths.get(
                        filePath.replace(psScript, psTemplateScript)
                ),
                Paths.get(
                        filePath
                ),
                StandardCopyOption.REPLACE_EXISTING
        );

        this.replaceInFile(
                filePath,
                "%%PROMPT%%",
                prompt
        );

        ProcessBuilder pbuilder = new ProcessBuilder(
                "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        "\"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; powershell -File .\\ollama.ps1\""
                );

        pbuilder.directory(new File(currentPath)); // set working directory to run this command
        pbuilder.redirectErrorStream(false); // keep stdout and stderr separate

        List<ProcessBuilder> builders = Arrays.asList(pbuilder);

        List<Process> processes = startPipeline(builders);
        Process process = processes.getLast();

        List<String> response = new ArrayList<>(new ArrayList<>(List.of()));

        try {
            try (BufferedReader stdOut = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
                 BufferedReader stdErr = new BufferedReader(
                         new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {

                String text;

                String line;
                System.out.println("----- OUTPUT -----");

                while ((text = stdOut.readLine()) != null) {
                    int positionStart = text.indexOf("response", 0);

                    if (positionStart != -1) {
                        //int positionEnd = text.indexOf(",\"done\":", positionStart) + ": ".length();
                        int positionEnd = text.indexOf(": ", positionStart) + ": ".length();

                        if (positionEnd != -1) {
                            String str = text.substring(
                                    //positionStart + "\"response\":".length(),
                                    //positionEnd - ",".length()
                                    //positionStart + "response".length(),
                                    //positionEnd - "".length()
                                    positionEnd
                            ).replaceAll(
                                    "(^\")|(\",$)",
                                    ""
                            ); // this last replaceAll removes single quote from the
                            // beginning and end of string

                            if (!str.isEmpty()) {
                                response.add(str);
                            }
                        }
                    } else {

                        String ending = "done                 : ";
                        positionStart = text.indexOf(ending, 0);

                        if (positionStart != -1) {
                            break; // exit for
                        } else {
                            positionStart = 0;
                            String str = text.substring(
                                    positionStart
                            ).replaceAll(
                                    "(^\")|(\",$)",
                                    ""
                            ); // this last replaceAll removes single quote from the
                            // beginning and end of string

                            str = StringUtils.stripEnd(str, " ");
                            if (!str.isEmpty()) { response.add(str); }
                        }
                    }
                } // exit while

                if (response.isEmpty()) {
                    response.add("There is not response.");
                }

                System.out.println("------ ERRORS ------");

                while ((line = stdErr.readLine()) != null) {
                    System.err.println(line);
                }
            } // exit try

            int exitCode = process.waitFor();
            System.out.println("Process exited with code: " + exitCode);
        } catch (IOException e) {
            System.err.println("I/O Error: " + e.getMessage());
            response.clear();
            response.add(e.getMessage()); // set error description
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Process interrupted");
        } // exit try

        this.prompt = prompt;
        this.response = response;
    }

    private List<String> readOutput(InputStream inputStream) {
        Reader isr = new InputStreamReader(inputStream);
        BufferedReader r = new BufferedReader(isr);
        return r.lines().toList();
    }

    private void replaceInFile(String filePath, String target, String replacement) throws IOException {
        Path path = Paths.get(filePath);

        if (!Files.exists(path)) {
            throw new IOException("File not found: " + filePath);
        }

        Files.writeString(
                path,
                Files
                        .readString(path, StandardCharsets.UTF_8)
                        .replace(target, replacement),
                StandardCharsets.UTF_8,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }
}
