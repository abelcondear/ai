package com.ollama.ollama.component;

import com.ollama.ollama.configuration.TaskStatus;
import com.ollama.ollama.model.TrackedEvent;
import com.ollama.ollama.utils.OllamaReader;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import java.io.IOException;

@Component
public class TrackedEventListener {

    private final TaskStatusStore statusStore;
    private final TaskPromptStore promptStore;

    public TrackedEventListener(TaskStatusStore statusStore, TaskPromptStore promptStore) {
        this.statusStore = statusStore;
        this.promptStore = promptStore;
    }

    @Async
    @EventListener
    public void handleTrackedEvent(TrackedEvent event) throws IOException {
        String taskId = event.getTaskId();

        statusStore.addStatus(taskId, TaskStatus.RUNNING);
        System.out.println("Processing task " + taskId);

        Thread thread = new Thread(() -> {
            try {
                OllamaReader ollamaReader = new OllamaReader(event.getTaskPrompt().getPrompt());

                event.getTaskPrompt().setResponse(ollamaReader.response);
                promptStore.addPrompt(taskId, event.getTaskPrompt());
                statusStore.addStatus(taskId, TaskStatus.COMPLETED);

                System.out.println("Task " + taskId + " completed.");
            } catch (IOException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }
        });

        thread.start();
    }
}