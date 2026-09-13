# OS Tasks

This repository contains Java implementations of multithreading concepts as part of the Operating Systems assignment.

## Tasks

### Task 1: Producer-Consumer Problem

Implemented the Producer-Consumer problem using Java threads.

The program demonstrates:
- Producer and Consumer threads
- Bounded buffer
- Thread synchronization
- `wait()` and `notifyAll()`
- Handling of full and empty buffer conditions

**File:** `SmartConveyor.java`

---

### Task 2: Matrix Multiplication Using Threads

Implemented multiplication of two **100 × 100 matrices** using Java Virtual Threads.

The program demonstrates:
- Java Virtual Threads
- One thread for each scalar multiplication operation
- Matrix multiplication of 100 × 100 matrices
- TensorFlow for result verification
- Swing-based animation to visualize the computation
- Execution of **1,000,000 multiplication operations**

**Files:**
- `MatrixMultiplication.java`
- `pom.xml`

## Technologies Used

- Java 21
- Java Virtual Threads
- TensorFlow
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
