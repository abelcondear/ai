package util;
// ----
import java.net.URISyntaxException;
// ----
import file.FileUtils;
// ----

public class Directory {
    public Directory() { /* TODO */ }

    public String getRFilePath(String project_name) throws URISyntaxException {
        FileUtils fileUtils = new FileUtils();

        // CSV exists as file source
        String path = fileUtils.GetFilePath("Elnino.csv");

        String directory =
                path.substring
                (
                    0,
                    path.lastIndexOf("\\")
                );

        // searching project name ("pmml4s")
        // 10 iterations as maximum iteration
        for (int x = 0; x < 10; x ++) {
            if (
                directory.indexOf
                (
                    "\\" + project_name + "\\",
                    directory.lastIndexOf("\\")
                ) != -1
            ) {
                directory = directory.substring
                        (
                            0,
                            directory.lastIndexOf("\\")
                        );
                break;
            }
        }

        return directory;
    }
}
