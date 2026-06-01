plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.scala-lang:scala-library:2.13.9")
    implementation("org.apache.commons:commons-lang3:3.18.0")
    implementation("org.pmml4s:pmml4s_3:1.5.8")
    implementation("org.rosuda.REngine:Rserve:1.8.1")
    implementation("org.rosuda.REngine:REngine:2.1.0")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}