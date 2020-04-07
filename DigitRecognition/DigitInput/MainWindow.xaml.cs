using System;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using DigitRecognizer;

namespace DigitInput
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private static readonly int _pixelCount = 28 * 28;
        private Point _currentPoint = new Point();

        public MainWindow()
        {
            InitializeComponent();
        }


        private void ResetCanvasButton_Click(object sender, RoutedEventArgs e)
        {
            PaintCanvas.Children.Clear();
        }

        private void DetectDigitButton_Click(object sender, RoutedEventArgs e)
        {
            var rtb = new RenderTargetBitmap((int)PaintCanvas.RenderSize.Width,
                (int)PaintCanvas.RenderSize.Height, 96d, 96d, PixelFormats.Default);
            rtb.Render(PaintCanvas);

            /*BitmapEncoder pngEncoder = new PngBitmapEncoder();
            pngEncoder.Frames.Add(BitmapFrame.Create(rtb));

            //save to memory stream
            System.IO.MemoryStream ms = new System.IO.MemoryStream();

            pngEncoder.Save(ms);
            ms.Close();
            System.IO.File.WriteAllBytes("logo.png", ms.ToArray());
            Console.WriteLine("Done");*/
            
            var pixels = new byte[_pixelCount * 4];
            rtb.CopyPixels(pixels, 112, 0);

            var pixelValues = new float[_pixelCount];
            for (var i = 0; i < _pixelCount; i++)
            {
                var bytes = pixels.Skip(i * 4).Take(4).Reverse().ToArray();
                var val = (float) BitConverter.ToInt32(bytes, 0);
                pixelValues[i] = val;
            }

            var digitInput = new DigitInputData
            {
                PixelValues = pixelValues
            };

            var digit = DigitRecognizer.DigitRecognizer.PredictDigit(digitInput);
            PredictionTextBox.Text = $"Predicted digit: {digit}";
        }

        private void PaintCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ButtonState == MouseButtonState.Pressed)
                _currentPoint = e.GetPosition(PaintCanvas);
        }

        private void PaintCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton != MouseButtonState.Pressed)
            {
                return;
            }

            var line = new Line
            {
                Stroke = new SolidColorBrush(Colors.Black),
                X1 = _currentPoint.X,
                Y1 = _currentPoint.Y,
                X2 = e.GetPosition(PaintCanvas).X,
                Y2 = e.GetPosition(PaintCanvas).Y,
                StrokeThickness = 3,
            };

            _currentPoint = e.GetPosition(PaintCanvas);

            PaintCanvas.Children.Add(line);
        }
    }
}
