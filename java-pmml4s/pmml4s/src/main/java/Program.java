import org.pmml4s.model.Model;
// ----
import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.lang.Process;
// ----
import file.FileUtils;
import model.ModelUtils;
import util.Directory;
import util.Printable;
import command.UserProcess;
// ----

public class Program {
    private final FileUtils fileUtils = new FileUtils();
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
        Directory dir = new Directory();
        String directory = dir.getRFilePath(project_name);

        // This is taken from .\build\resources\main\
        String filePath = directory + "\\ToPMML.R";

        // RTerm command path
        String commandPath = "C:\\Program Files\\" +
                            "R\\R-4.5.1\\bin\\x64\\" +
                            ".\\Rterm";

        // execute command
        UserProcess process = new UserProcess();
        Process p = process.Run(commandPath, filePath);

        Printable printOutput = new Printable();

        // Wait process to be finished
        int exitCode = p.waitFor();
        printOutput.PrintExitCode(exitCode);

        // ----
        Model model = modelUtils.createModel("Elnino.pmml");
        Map<String, Double> values = modelUtils.createValues();

        // ----
        double predicted = modelUtils.getRegressionValue(model, values);
        printOutput.PrintPredictedValue(predicted);
    }
}