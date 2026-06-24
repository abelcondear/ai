// ----
package util;
// ----

public class Printable {
    public Printable(){ /* TODO */ }

    public void PrintExitCode(int value) {
        System.out.println
        (
            " -------------------------- "
        );
        System.out.println
        (
            "Process exit code:" +
            String.format(" %d", value)
        );
        System.out.println
        (
            " -------------------------- "
        );
    }

    public void PrintMessageValue(String value) {
        System.out.println
        (
            " -------------------------- "
        );
        System.out.println("Message: ");
        System.out.println
        (
            String.format(" %s", value)
        );
        System.out.println
        (
            " -------------------------- "
        );
        System.out.println();
    }

    public void PrintPredictedValue(double value) {
        System.out.println
        (
            " -------------------------- "
        );
        System.out.println(" Predicted value: ");
        System.out.println
        (
            String.format(" %.10f", value)
        );
        System.out.println
        (
            " -------------------------- "
        );
        System.out.println();
    }
}
