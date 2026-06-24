// ----
package file;
// ----
import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
// ----

public class FileUtils {
    public FileUtils() {
        //TODO
    }

    public String GetFilePath(String filename)
            throws URISyntaxException
    {
        URL resource = FileUtils
                .class
                .getClassLoader()
                .getResource(filename);

        String path;

        if (resource != null) {
            File file = new File(resource.toURI());
            path = file.getAbsolutePath();
        }
        else {
            path = "";
        }

        return path;
    }
}
