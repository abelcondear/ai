// ----
package util;
// ----
import java.util.Locale;
// ----

public class DetectOS {
    public DetectOS() { /* TODO */  }

    private static final String OS = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);

    private static final boolean IS_WINDOWS = OS.contains("win");
    private static final boolean IS_LINUX = OS.contains("nux") || OS.contains("nix") || OS.contains("aix");
    private static final boolean IS_MAC = OS.contains("mac");

    public static boolean isWindows() {
        return IS_WINDOWS;
    }

    public static boolean isLinux() {
        return IS_LINUX;
    }

    public static boolean isMac() {
        return IS_MAC;
    }
}
