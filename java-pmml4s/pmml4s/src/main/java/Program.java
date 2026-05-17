import org.pmml4s.model.Model;
import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.*;

public class Program {
    private final Model model =
            Model.fromFile(
                    new File(
                        GetFilePath("Elnino.pmml")
                    )
            );

    public Program() throws URISyntaxException {
        //TODO
    }

    public FileOutputStream saveStream(InputStream is, File targetFile) throws IOException {
        FileOutputStream result;

        try (FileOutputStream fos = new FileOutputStream(targetFile)) {
            is.transferTo(fos);
            result = fos;
        }

        return result;
    }

    public String GetFilePath(String filename) throws URISyntaxException {
        URL resource = Program
                        .class
                        .getClassLoader()
                        .getResource(filename);
        String path;

        if (resource != null) {
            File file = new File(resource.toURI());
            path = file.getAbsolutePath();
            return path;
        }
        else {
            return "";
        }
    }

    public Double getRegressionValue(Map<String, Double>
                                             values) {
        Object[] valuesMap =
                Arrays.stream(model.inputNames())
                        .map(values::get)
                        .toArray();

        Object[] result = model.predict(valuesMap);

        return (Double) result[0];
    }

    public void main(String[] args) {
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
        "s_s_temp",  28.78d
        );

        double predicted = getRegressionValue(values);

        System.out.println(" -------------------------- ");
        System.out.println(" Predicted value: ");
        System.out.println(String.format(" %.10f", predicted));
        System.out.println(" -------------------------- ");
        System.out.println();

//      Output Console
//      --------------------------
//       Predicted value:
//       29,2236885246
//      --------------------------

    }
}