import org.pmml4s.model.Model;
import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import misc.Utils;

public class Program {
    private final Utils utils = new Utils();

    public Program() throws URISyntaxException {
        //TODO
    }

    public void main(String[] args) throws
            IOException,
            URISyntaxException, InterruptedException {

        // CSV exists as file source
        String path = utils.GetFilePath("Elnino.csv");

        String directory = path.substring(
                                0,
                                path.lastIndexOf("\\")
                            );

        String project_name = "pmml4s";

        // searching project name ("pmml4s")
        // 10 iterations as maximum iteration
        for (int x = 0; x < 10; x ++) {
            if (
                    directory.indexOf(
                            "\\" + project_name + "\\",
                            directory.lastIndexOf("\\")
                    ) != -1
            ) {
                directory = directory.substring(
                    0,
                    directory.lastIndexOf("\\")
                );
                break;
            }
        }

        // This is taken from .\build\resources\main\
        String filePath = directory + "\\ToPMML.R";

        String commandPath = "c:\\Program Files\\" +
                            "R\\R-4.5.1\\bin\\x64\\" +
                            ".\\Rterm";

        ProcessBuilder pb = new ProcessBuilder(
                commandPath, // command
                "-f", // file
                "\"" + filePath + "\"" // file path to be executed
        );
        pb.inheritIO();
        Process p = pb.start();

        // Wait process to be finished
        int exitCode = p.waitFor();

        System.out.println(" -------------------------- ");
        System.out.println("Process exit code:" + String.format(" %d", exitCode));
        System.out.println(" -------------------------- ");

        // ----
        Model model = Model.fromFile(
                new File(
                        utils.GetFilePath("Elnino.pmml")
                )
        );

        Map<String, Double> values = Map.of(
            "buoy_day_ID", 776d,
            "buoy", 59d,
            "day", 7d,
            "latitude", -8.03d,
            "longitude", 164.82d,
            "zon_winds", 0d,
            "mer_winds", 0d,
            "humidity", 89.50d,
            "airtemp", 27.51d,
            "s_s_temp",  0d
        );

        double predicted = utils.getRegressionValue(model, values);

        System.out.println(" -------------------------- ");
        System.out.println(" Predicted value: ");
        System.out.println(String.format(" %.10f", predicted));
        System.out.println(" -------------------------- ");
        System.out.println();

//      ::: Output Console :::
//      --------------------------
//       Predicted value:
//       real value: 29,2236885246
//       s_s_temp (k10): 29.22d
//      --------------------------

    }
}