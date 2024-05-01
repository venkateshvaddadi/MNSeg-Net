<h1>MN-Net for Real-Time Median Nerve Segmentation in Ultrasonography</h1>
<h1>MN-Net Architecture</h1>
The CNN architecture of the proposed \textbf{MN-Net}, known as the Median Nerve Segmentation Network, is introduced. MN-Net is an efficient, lightweight network architecture; in this, there is a primary network comprising a UNet-based encoder-decoder structure, referred to as the main network. Alongside this main network, a subnetwork is integrated within it, learning concurrently with the main network and serving as the refinement network module.
<img src="images/Architecture_updated.png" alt="spinet-QSM architecture" width=100% height=100%>
<h1>Clinical setup for Real-Time Median Nerve Segmentation</h1>
The proposed model was deployed and made available as an End-to-End deep learning based software tool for Real-time checking in the clinical environment. It provides a parallel screen to the original US screen, which can show the US frame with a segmented MedianNerve along with its CSA.
<img src="images/Real_time_setup.png" alt="spinet-QSM architecture" width=100% height=100%>

