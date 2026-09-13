# OS Tasks

This repository contains Java implementations of multithreading concepts as part of the Operating Systems assignment.

## Tasks

### Task 1: Producer-Consumer Problem

Implemented the Producer-Consumer problem using Java threads.

The program demonstrates:
- Producer and Consumer threads
- A bounded circular buffer
- Thread synchronization
- `synchronized` methods
- `wait()` and `notifyAll()`
- Handling of full and empty buffer conditions
- Safe communication between threads

**File:** `SmartConveyor.java`

---

### Task 2: Matrix Multiplication Using Threads

Implemented multiplication of two **100 × 100 matrices** using Java Virtual Threads.

The program demonstrates:
- Java Virtual Threads
- One virtual thread for each individual scalar multiplication
- 100 × 100 matrix multiplication
- A total of 1,000,000 multiplication operations
- TensorFlow for independent result verification
- Swing-based animation to visualize the computation
- Atomic counters for tracking completed operations

**Files:**
- `MatrixMultiplication.java`
- `pom.xml`

## Technologies Used

- Java 21
- Java Virtual Threads
- TensorFlow Java
- Maven
- Java Swing

## Project Structure

```text
OS_Tasks/
│
├── src/
│   └── main/
│       └── java/
│           ├── SmartConveyor.java
│           └── MatrixMultiplication.java
│
└── pom.xml
