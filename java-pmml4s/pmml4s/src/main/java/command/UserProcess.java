package command;
// ----
import java.io.IOException;
// ----

public class UserProcess {
    public UserProcess() { /* TODO */ }

    public java.lang.Process Run(String commandPath, String filePath) throws IOException {
        ProcessBuilder pb = new ProcessBuilder
                (
                        commandPath, // command
                        "-f", // file
                        "\"" + filePath + "\"" // R file script path to be executed
                );
        pb.inheritIO();
        java.lang.Process p = pb.start();

        return p;
    }
}
