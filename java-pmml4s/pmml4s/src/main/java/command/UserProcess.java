// ----
package command;
// ----
import java.io.IOException;
import java.lang.Process;
// ----
import util.DetectOS;
// ----


public class UserProcess {
    public UserProcess() { /* TODO */ }

    public Process Run(String commandPath, String filePath) throws IOException {
        ProcessBuilder pb;

        if (DetectOS.isLinux()) {
            pb = new ProcessBuilder
                    (
                        commandPath, // command
                        String.format("%s", filePath) // R file script path to be executed
                    );
        }
        else {
            pb = new ProcessBuilder
                    (
                        commandPath, // command
                        "-f", // file
                        String.format("\"%s\"", filePath) // R file script path to be executed
                    );
        }

        pb.inheritIO();

        return (pb.start());
    }
}