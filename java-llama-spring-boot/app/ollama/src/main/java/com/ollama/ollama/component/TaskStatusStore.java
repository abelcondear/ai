package com.ollama.ollama.component;

import com.ollama.ollama.configuration.TaskStatus;
import org.springframework.stereotype.Component;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@Component
public class TaskStatusStore {
    // create array of statusMap
    private final ConcurrentMap<String, TaskStatus> statusMap = new ConcurrentHashMap<>();

    public TaskStatusStore() {
        //TODO
    }

    public void addStatus(String taskId, TaskStatus status) {
        statusMap.put(taskId, status);
    }

    public TaskStatus readStatus(String taskId) {
        return statusMap.get(taskId);
    }

    public void setStatus(String taskId, TaskStatus status) {
        statusMap.put(taskId, status);
    }

    public TaskStatus getStatus(String taskId) {
        return statusMap.getOrDefault(taskId, TaskStatus.PENDING);
    }
}