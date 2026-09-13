import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MatrixMultiplication {

    static final int SIZE = 100;

    static int[][] matrixA = new int[SIZE][SIZE];
    static int[][] matrixB = new int[SIZE][SIZE];
    static long[][] result = new long[SIZE][SIZE];

    static AtomicInteger multiplicationCount = new AtomicInteger(0);
    static AtomicInteger completedCells = new AtomicInteger(0);

    static volatile int activeRow = -1;
    static volatile int activeColumn = -1;
    static volatile boolean finished = false;

    static void initializeMatrices() {

        for (int i = 0; i < SIZE; i++) {

            for (int j = 0; j < SIZE; j++) {

                matrixA[i][j] = (i + 2 * j) % 9 + 1;
                matrixB[i][j] = (2 * i + j) % 9 + 1;
            }
        }
    }

    static void multiplyWithThreads() {

        for (int i = 0; i < SIZE; i++) {

            for (int j = 0; j < SIZE; j++) {

                activeRow = i;
                activeColumn = j;

                Thread[] workers = new Thread[SIZE];
                Object cellLock = new Object();

                for (int k = 0; k < SIZE; k++) {

                    final int row = i;
                    final int column = j;
                    final int position = k;

                    workers[k] = Thread.startVirtualThread(() -> {

                        long product =
                                (long) matrixA[row][position]
                                * matrixB[position][column];

                        synchronized (cellLock) {
                            result[row][column] += product;
                        }

                        multiplicationCount.incrementAndGet();
                    });
                }

                for (Thread worker : workers) {

                    try {
                        worker.join();

                    } catch (InterruptedException e) {

                        Thread.currentThread().interrupt();
                        return;
                    }
                }

                completedCells.incrementAndGet();

                try {
                    Thread.sleep(2);

                } catch (InterruptedException e) {

                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }

        activeRow = -1;
        activeColumn = -1;
        finished = true;
    }

    static long[][] tensorflowCalculation() {

        try (Graph graph = new Graph()) {

            TFloat32 tensorA =
                    TFloat32.tensorOf(
                            Shape.of(SIZE, SIZE)
                    );

            TFloat32 tensorB =
                    TFloat32.tensorOf(
                            Shape.of(SIZE, SIZE)
                    );

            for (int i = 0; i < SIZE; i++) {

                for (int j = 0; j < SIZE; j++) {

                    tensorA.setFloat(
                            matrixA[i][j],
                            i,
                            j
                    );

                    tensorB.setFloat(
                            matrixB[i][j],
                            i,
                            j
                    );
                }
            }

            org.tensorflow.op.Ops tf =
                    org.tensorflow.op.Ops.create(graph);

            var inputA =
                    tf.constant(tensorA);

            var inputB =
                    tf.constant(tensorB);

            var multiplication =
                    tf.linalg.matMul(
                            inputA,
                            inputB
                    );

            try (Session session =
                         new Session(graph)) {

                Tensor output =
                        session.runner()
                                .fetch(multiplication)
                                .run()
                                .get(0);

                TFloat32 outputData =
                        (TFloat32) output;

                long[][] tensorflowResult =
                        new long[SIZE][SIZE];

                for (int i = 0; i < SIZE; i++) {

                    for (int j = 0; j < SIZE; j++) {

                        tensorflowResult[i][j] =
                                Math.round(
                                        outputData.getFloat(
                                                i,
                                                j
                                        )
                                );
                    }
                }

                output.close();
                tensorA.close();
                tensorB.close();

                return tensorflowResult;
            }
        }
    }

    static boolean compareResults(
            long[][] tensorflowResult) {

        for (int i = 0; i < SIZE; i++) {

            for (int j = 0; j < SIZE; j++) {

                if (result[i][j]
                        != tensorflowResult[i][j]) {

                    return false;
                }
            }
        }

        return true;
    }

    static void displayResult() {

        System.out.println();
        System.out.println(
                "First 5 x 5 elements of Result Matrix"
        );

        System.out.println(
                "---------------------------------------"
        );

        for (int i = 0; i < 5; i++) {

            for (int j = 0; j < 5; j++) {

                System.out.printf(
                        "%8d",
                        result[i][j]
                );
            }

            System.out.println();
        }
    }

    static class MatrixPanel extends JPanel {

        static final int CELL_SIZE = 7;

        MatrixPanel() {

            setPreferredSize(
                    new Dimension(
                            SIZE * CELL_SIZE,
                            SIZE * CELL_SIZE
                    )
            );
        }

        @Override
        protected void paintComponent(
                Graphics g) {

            super.paintComponent(g);

            int completed =
                    completedCells.get();

            for (int i = 0; i < SIZE; i++) {

                for (int j = 0; j < SIZE; j++) {

                    int position =
                            i * SIZE + j;

                    if (position < completed) {

                        int value =
                                (int) Math.abs(
                                        result[i][j] % 180
                                );

                        g.setColor(
                                new Color(
                                        40 + value / 5,
                                        100 + value / 2,
                                        210 - value / 6
                                )
                        );

                    } else {

                        g.setColor(
                                new Color(
                                        235,
                                        235,
                                        235
                                )
                        );
                    }

                    g.fillRect(
                            j * CELL_SIZE,
                            i * CELL_SIZE,
                            CELL_SIZE - 1,
                            CELL_SIZE - 1
                    );
                }
            }

            if (!finished && activeRow >= 0) {

                g.setColor(Color.RED);

                g.drawRect(
                        activeColumn * CELL_SIZE,
                        activeRow * CELL_SIZE,
                        CELL_SIZE - 1,
                        CELL_SIZE - 1
                );
            }
        }
    }

    static void createAnimation() {

        JFrame window =
                new JFrame(
                        "100 x 100 Matrix Multiplication"
                );

        window.setDefaultCloseOperation(
                JFrame.EXIT_ON_CLOSE
        );

        window.setLayout(
                new BorderLayout()
        );

        JLabel heading =
                new JLabel(
                        "Matrix Multiplication Using Threads",
                        SwingConstants.CENTER
                );

        heading.setFont(
                new Font(
                        "Arial",
                        Font.BOLD,
                        20
                )
        );

        window.add(
                heading,
                BorderLayout.NORTH
        );

        MatrixPanel panel =
                new MatrixPanel();

        window.add(
                panel,
                BorderLayout.CENTER
        );

        JPanel information =
                new JPanel();

        information.setLayout(
                new BoxLayout(
                        information,
                        BoxLayout.Y_AXIS
                )
        );

        JLabel status =
                new JLabel(
                        "Status: Processing..."
                );

        JLabel cells =
                new JLabel(
                        "Completed cells: 0 / 10000"
                );

        JLabel operations =
                new JLabel(
                        "Multiplications: 0 / 1000000"
                );

        JLabel current =
                new JLabel(
                        "Current cell: --"
                );

        information.add(status);
        information.add(cells);
        information.add(operations);
        information.add(current);

        window.add(
                information,
                BorderLayout.SOUTH
        );

        Timer timer =
                new Timer(
                        100,
                        event -> {

                            cells.setText(
                                    "Completed cells: "
                                    + completedCells.get()
                                    + " / 10000"
                            );

                            operations.setText(
                                    "Multiplications: "
                                    + multiplicationCount.get()
                                    + " / 1000000"
                            );

                            if (activeRow >= 0) {

                                current.setText(
                                        "Current cell: C["
                                        + activeRow
                                        + "]["
                                        + activeColumn
                                        + "]"
                                );

                            } else {

                                current.setText(
                                        "Current cell: Completed"
                                );
                            }

                            if (finished) {

                                status.setText(
                                        "Status: Calculation Completed"
                                );

                                ((Timer) event.getSource())
                                        .stop();
                            }

                            panel.repaint();
                        }
                );

        timer.start();

        window.pack();

        window.setMinimumSize(
                new Dimension(
                        800,
                        850
                )
        );

        window.setLocationRelativeTo(null);

        window.setVisible(true);
    }

    public static void main(String[] args) {

        System.out.println(
                "=============================================="
        );

        System.out.println(
                "       MATRIX MULTIPLICATION USING THREADS"
        );

        System.out.println(
                "=============================================="
        );

        System.out.println(
                "Matrix A              : 100 x 100"
        );

        System.out.println(
                "Matrix B              : 100 x 100"
        );

        System.out.println(
                "Result Matrix         : 100 x 100"
        );

        System.out.println(
                "Thread Technology     : Java Virtual Threads"
        );

        System.out.println(
                "Framework             : TensorFlow"
        );

        System.out.println(
                "Total Multiplications : 1,000,000"
        );

        initializeMatrices();

        SwingUtilities.invokeLater(
                MatrixMultiplication::createAnimation
        );

        try {

            Thread.sleep(300);

        } catch (InterruptedException e) {

            Thread.currentThread().interrupt();
            return;
        }

        long startTime =
                System.nanoTime();

        multiplyWithThreads();

        long endTime =
                System.nanoTime();

        double executionTime =
                (endTime - startTime)
                / 1_000_000_000.0;

        System.out.println();

        System.out.println(
                "Thread-based multiplication completed."
        );

        System.out.println(
                "Completed operations: "
                + multiplicationCount.get()
        );

        System.out.println(
                "Completed cells: "
                + completedCells.get()
        );

        System.out.printf(
                "Execution time: %.3f seconds%n",
                executionTime
        );

        System.out.println();

        System.out.println(
                "Verifying result using TensorFlow..."
        );

        long[][] tensorflowResult =
                tensorflowCalculation();

        boolean correct =
                compareResults(
                        tensorflowResult
                );

        System.out.println(
                "TensorFlow verification: "
                + (correct
                        ? "PASSED"
                        : "FAILED")
        );

        displayResult();

        System.out.println();

        System.out.println(
                "=============================================="
        );

        System.out.println(
                "             PROGRAM COMPLETED"
        );

        System.out.println(
                "=============================================="
        );
    }
}
