package misc;

import org.pmml4s.model.Model;
import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;
import java.util.Map;

public class Utils {
    public Utils() {
        //TODO
    }

    public String GetFilePath(String filename)
            throws URISyntaxException
    {
        URL resource = Utils
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

    public Double getRegressionValue
    (
        Model model,
        Map<String, Double> values
    ) {
        Object[] valuesMap =
                Arrays.stream
                        (
                            model.inputNames()
                        )
                        .map(values::get)
                        .toArray();

        Object[] result = model.predict(valuesMap);

        return (Double) result[0];
    }
}
