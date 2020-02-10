using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;
using FreeImageAPI;

namespace GroudtruthBuilder
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //FREE_IMAGE_FORMAT fif_format = FREE_IMAGE_FORMAT.FIF_JP2;
            //Bitmap mybp = FreeImage.LoadBitmap(@"C:\Users\green\Downloads\sn87062082-18790611.jp2", 
            //    FREE_IMAGE_LOAD_FLAGS.JPEG_ACCURATE,
            //    ref fif_format);
            //Size s = mybp.Size;

            XmlDocument doc = new XmlDocument();
            doc.Load(@"G:\DC_intern\pilotprojdc\GroudtruthBuilder\GroudtruthBuilder\my.xml");
            StringBuilder sb = new StringBuilder();
            foreach (XmlNode node in doc.GetElementsByTagName("TextBlock"))
            {
                string tmp = "rectangle('Position', [{0},{1},{2},{3}], 'LineWidth', 2, 'LineStyle','--');";
                string xpos, ypos, width, height;
                xpos = node.Attributes.GetNamedItem("HPOS").Value;
                ypos = node.Attributes.GetNamedItem("VPOS").Value;
                width = node.Attributes.GetNamedItem("WIDTH").Value;
                height = node.Attributes.GetNamedItem("HEIGHT").Value;
                sb.Append(String.Format(tmp, xpos, ypos, width, height)+"\n");
            }
            string matlab = sb.ToString();
            var x = doc.GetElementsByTagName("TextBlock");
            

        }
    }
}
