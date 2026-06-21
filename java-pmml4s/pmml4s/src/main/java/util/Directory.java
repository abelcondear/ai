// ----
package util;
// ----
import java.net.URISyntaxException;
import java.util.Locale;
// ----
import file.FileUtils;
import util.DetectOS;
// ----

public class Directory {
    public Directory() { /* TODO */ }

    public String getRFilePath(String project_name) throws URISyntaxException {
        FileUtils fileUtils = new FileUtils();

        // CSV exists as file source
        String path = fileUtils.GetFilePath("Elnino.csv");

        String slash =  (DetectOS.isLinux()) ? "/":"\\";

        String directory =
                path.substring
                        (
                            0,
                            path.lastIndexOf(slash)
                        );

        // searching project name ("pmml4s")
        // 10 iterations as maximum iteration
        for (int x = 0; x < 10; x ++) {
            if (
                    directory.indexOf
                            (
                                slash + project_name + slash,
                                directory.lastIndexOf(slash)
                            ) != -1
            ) {
                directory = directory.substring
                        (
                            0,
                            directory.lastIndexOf(slash)
                        );
                break;
            }
        }

        return directory;
    }
}
