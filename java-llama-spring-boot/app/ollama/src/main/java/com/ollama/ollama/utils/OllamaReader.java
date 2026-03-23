package com.ollama.ollama.utils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static com.ollama.ollama.component.ApplicationProperties.AppName;
import static java.lang.ProcessBuilder.startPipeline;

import com.ollama.ollama.error.ShellExecutionException;
import org.apache.commons.lang3.StringUtils;
import java.io.IOException;

public class OllamaReader {
    public String prompt;
    public List<String> response;
    public List<String> error;

    public OllamaReader(String prompt) throws RuntimeException, IOException, InterruptedException {
        String path = new File(".").getAbsolutePath();

        String appNameDir = "\\" + AppName + "\\";

        String psScript = AppName + ".ps1";
        String psTemplateScript = AppName + ".template.ps1";

        int index = path.indexOf(appNameDir);

        String currentPath = path.substring(0, index + appNameDir.length());
        String filePath = currentPath + psScript;

        String commandPs = String.format(
                "\"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; powershell -File '%s' -Z '%s'\"",
                psTemplateScript,
                prompt
        );

        ProcessBuilder pbuilder = new ProcessBuilder(
                "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        commandPs
            );

        pbuilder.directory(new File(currentPath)); // set working directory to run this command
        pbuilder.redirectErrorStream(false); // keep stdout and stderr separate

        List<ProcessBuilder> builders = Arrays.asList(pbuilder);

        List<Process> processes = startPipeline(builders);
        Process process = processes.getLast();

        List<String> response = new ArrayList<>(new ArrayList<>(List.of()));
        List<String> error = new ArrayList<>(new ArrayList<>(List.of()));

        try {
            try (
                 BufferedReader stdOut = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8)
                 );
                 BufferedReader stdErr = new BufferedReader(
                    new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8)
                 )
            ) {
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
                            break; // exit for // reading of response is finished
                        } else if (isReadingResponse) {
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
                    //System.err.println(line); // printing in console::omitted
                    if (!errorFound) { errorFound = true; }
                    errorDescription.add(line);
                }

                if (errorFound) {
                    Thread.currentThread().interrupt();
                    error.add("Sorry. Response could not be reached by AI.");
                    throw new ShellExecutionException(
                            String.join(
                                    "\n",
                                    errorDescription
                            )
                    );
                }
                else if (response.isEmpty()) {
                    response.add("There is not response.");
                }
            } catch (ShellExecutionException e) {
                Thread.currentThread().interrupt();
                throw e; // Re-Throwing error
            } catch (Exception e) { // general exception
                Thread.currentThread().interrupt();
                throw e; // Re-Throwing error
            } // exit try // this try-catch result will be caught by the outer try-catch

            int exitCode = process.waitFor();
            System.out.println("Process exited with code: " + exitCode);
        } catch (IOException e) {
            Thread.currentThread().interrupt();
            System.err.println("I/O Error. " + e.getMessage());
            System.err.println("IOException. Process interrupted");
            error.clear();
            error.add(e.getMessage()); // set error description
            throw e; // Re-Throw error
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Interrupted Error. " + e.getMessage());
            System.err.println("InterruptedException. Process interrupted");
            error.clear();
            error.add(e.getMessage()); // set error description
            throw e; // Re-Throw error
        } catch (ShellExecutionException e) {
            Thread.currentThread().interrupt();
            System.err.println("ShellExecution Error. " + e.getMessage());
            System.err.println("ShellExecutionException. Process interrupted");
            error.clear();
            error.add(e.getMessage()); // set error description
            throw e; // Re-Throw error
        } catch (Exception e) { // general exception
            Thread.currentThread().interrupt();
            System.err.println("Exception Error. " + e.getMessage());
            System.err.println("Exception. Process interrupted");
            error.clear();
            error.add(e.getMessage()); // set error description
            throw e; // Re-Throw error
        } // exit try

        this.prompt = prompt;
        this.response = response;
        this.error = error;
    }

    private List<String> readOutput(InputStream inputStream) {
        Reader isr = new InputStreamReader(inputStream);
        BufferedReader r = new BufferedReader(isr);
        return r.lines().toList();
    }
}
