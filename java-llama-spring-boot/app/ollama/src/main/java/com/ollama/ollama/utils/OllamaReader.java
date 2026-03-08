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
import java.lang.RuntimeException;
import com.ollama.ollama.error.ShellExecutionException;

public class OllamaReader {
    public String prompt;
    public List<String> response;

    public OllamaReader(String prompt) throws RuntimeException, IOException {
        String path = new File(".").getAbsolutePath();

        String appNameDir = "\\" + AppName + "\\";

        String psScript = AppName + ".ps1";
        String psTemplateScript = AppName + ".template.ps1";

        int index = path.indexOf(appNameDir);

        String currentPath = path.substring(0, index + appNameDir.length());
        String filePath = currentPath + psScript;

        this.restoreShellScript(filePath, psScript, psTemplateScript, prompt);

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

                boolean isReadingResponse = false;

                while ((text = stdOut.readLine()) != null) {
                    int positionStart = text.indexOf("response", 0); // from the beginning
                    int positionEnd = 0;

                    if (positionStart == 0) { // should be at first position of the string
                        isReadingResponse = true;
                        positionEnd = text.indexOf(": ", positionStart + "response".length());

                        if (positionEnd != -1) {
                            String str = text.substring(
                                    positionEnd + ": ".length()
                            ).replaceAll(
                                    "(^\")|(\",$)",
                                    ""
                            );
                            // this last replaceAll removes single quote from the
                            // beginning and end of string

                            if (!str.isEmpty()) { // string should have content before adding to response
                                response.add(str);
                            }
                        }
                    } else {
                        String ending = "done";
                        positionStart = text.indexOf(ending, 0);

                        if (positionStart == 0) { // should be at first position of the string
                            break; // exit for
                        } else if (isReadingResponse) {  // only do the follow steps when the string format is like "key: value"
                            positionStart = 0;

                            String str = text.substring(
                                    positionStart
                            ).replaceAll(
                                    "(^\")|(\",$)",
                                    ""
                            );
                            // this last replaceAll removes single quote from the
                            // beginning and end of string

                            str = StringUtils.strip(str, " "); // remove spaces from both sides
                            if (!str.isEmpty()) { // string should have content before adding to response
                                response.add(str);
                            }
                        }
                    }
                } // exit while

                System.out.println("------ ERRORS ------");

                boolean errorFound = false;
                List<String> errorDescription= new ArrayList<>(new ArrayList<>(List.of()));

                while ((line = stdErr.readLine()) != null) {
                    System.err.println(line);
                    errorFound = true;
                    errorDescription.add(line);
                }

                if (errorFound) {
                    response.add("Sorry. Response could be reached by AI. An error occurred.");
                    throw new ShellExecutionException(String.join("\n", errorDescription));
                }
                else if (response.isEmpty()) {
                    response.add("There is not response.");
                }
            } // exit try

            int exitCode = process.waitFor();
            System.out.println("Process exited with code: " + exitCode);
        } catch (IOException e) {
            System.err.println("I/O Error: " + e.getMessage());
            response.clear();
            response.add(e.getMessage()); // set error description in response
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Process interrupted");
        } // exit try

        this.prompt = prompt;
        this.response = response;
    }

    private void restoreShellScript(String filePath, String psScript, String psTemplateScript, String prompt) throws RuntimeException, IOException {
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
