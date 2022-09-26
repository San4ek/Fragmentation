import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat img = Imgcodecs.imread("C:\\image\\src\\" +
                "grass.jpg");
        if (img.empty()) {
            System.out.println("Не удалось загрузить изображение");
            return;
        }

        HighGui.imshow("Оригинал",img );
        HighGui.waitKey();
        Mat data = img.reshape(1, img.rows() * img.cols() * img.channels());
        data.convertTo(data, CvType.CV_32F, 1.0 / 255);
        Mat bestLabels = new Mat();
        Mat centers = new Mat();
        TermCriteria criteria = new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 10, 1);

        int K = 3;
        Core.kmeans(data, K, bestLabels, criteria, 5, Core.KMEANS_RANDOM_CENTERS, centers);
        Mat colors = new Mat();
        centers.t().convertTo(colors, CvType.CV_8U, 255);
        Mat lut = new Mat(1, 256, CvType.CV_8UC1, new Scalar(0));
        colors.copyTo(new Mat(lut, new Range(0, 1), new Range(0, colors.cols())));
        Mat result = bestLabels.reshape(img.channels(), img.rows());
        result.convertTo(result, CvType.CV_8U);
        Core.LUT(result, lut, result);
        Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
        HighGui.imshow("Результат = "+K, result);
        HighGui.waitKey();
        img.release(); data.release(); result.release();
        bestLabels.release(); centers.release();
        colors.release(); lut.release();

    }
}