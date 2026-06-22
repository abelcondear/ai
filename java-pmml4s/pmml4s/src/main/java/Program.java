// ----
import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.lang.Process;
// ----
import model.ModelUtils;
import util.Directory;
import util.Printable;
import util.DetectOS;
import command.UserProcess;
// ----

public class Program {
    private final ModelUtils modelUtils = new ModelUtils();

    public Program() { /* TODO */ }

    public void main(String[] args) throws
        IOException,
        URISyntaxException,
        InterruptedException
    {
        // ----
        String project_name = "pmml4s";
        // ----

        // ----
        Directory dir = new Directory();
        String directory = dir.getRFilePath(project_name);
        // ----

        // ----
        Printable printOutput = new Printable();

        if (!DetectOS.isLinux() && !DetectOS.isWindows()) {
            // ----
            printOutput.PrintMessageValue
            (
                "Program can be only executed under Windows or Linux environment."
            );
            // ----                        
        }
        else {
            // ----
            UserProcess process = new UserProcess();

            Process p;
            String commandLine = "";
            String absolutePath = "";

            if (DetectOS.isLinux()) {
                // ----
                commandLine = "/bin/Rscript";
                absolutePath = directory + "/ToPMML.R";

                p = process.Run
                        (
                            commandLine,
                            absolutePath
                        );
                // ----

                // Wait process to be finished ----
                int exitCode = p.waitFor();
                printOutput.PrintExitCode(exitCode);
                // ----
            }
            else if (DetectOS.isWindows()) {
                // ----
                commandLine = "C:\\Program Files\\" +
                        "R\\R-4.5.1\\bin\\x64\\" +
                        ".\\Rterm";
                absolutePath = directory + "\\ToPMML.R";

                p = process.Run
                        (
                            commandLine,
                            absolutePath
                        );
                // ----

                // Wait process to be finished ----
                int exitCode = p.waitFor();
                printOutput.PrintExitCode(exitCode);
                // ----
            }

            // ----
            double predicted = modelUtils.getRegressionValue
                    (
                        modelUtils.createModel("Elnino.pmml"),
                        modelUtils.createValues()
                    );
            printOutput.PrintPredictedValue(predicted);
            // ----
        }
    }
}