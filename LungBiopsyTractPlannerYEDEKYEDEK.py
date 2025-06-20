"""
Lung Biopsy Tract Planner Module

This module calculates safe biopsy tracts and analyzes risks such as hemorrhage and pneumothorax. 
Risk factors and coefficients are derived from the following academic references:
- Hemorrhage risks (lesion location, size, depth): Zhu J et al., 2020
- Pneumothorax risks (bullae, emphysema in the target lobe): Huo YR et al., 2020
- Pleural effusion risk reduction factors: Brönnimann MP et al., 2024

These references provide a robust foundation for accurate risk analysis and decision-making during lung biopsy procedures.
"""
# Import standard libraries
from __future__ import annotations
import sys, io, getpass, logging, os, time, subprocess, math, random, traceback, queue, qt, threading, slicer, glob, json, xmltodict, vtk, numpy as np, nibabel as nib
from vtk import vtkPointLocator, vtkPolyDataNormals, vtkPoints, vtkIdList, vtkPolyData, vtkOBBTree, vtkImplicitBoolean, vtkImplicitPolyDataDistance
from vtk.util.numpy_support import vtk_to_numpy
from collections import defaultdict
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Annotated
from pathlib import Path
BASE_DIR = Path.home() / "SlicerTemp"
BASE_DIR.mkdir(exist_ok=True)
TEMP_DIR   = BASE_DIR
OUTPUT_DIR = BASE_DIR / "TotalSegmentatorOutput"

from totalsegmentator.nifti_ext_header import load_multilabel_nifti
vtk.vtkObject.GlobalWarningDisplayOn() 
_thread_local = threading.local()
_cachedClosedSurf = {}
def _getClosed(segNode, segId):
    key = (segNode.GetID(), segId)
    if key not in _cachedClosedSurf:
        raw = vtk.vtkPolyData()
        segNode.GetClosedSurfaceRepresentation(segId, raw)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(raw)
        clean.Update()
        _cachedClosedSurf[key] = clean.GetOutput()   # temiz hâlini sakla
    return _cachedClosedSurf[key]    
    
_cachedImplicitVessel = {}

def _getVesselImplicit(segNode):
    """
    VESSEL_NAMES birleşimini temsil eden vtkImplicitBoolean objesini
    segNode + mtime anahtarına göre önbelleğe koyar.
    """
    key = (segNode.GetID(), segNode.GetMTime())
    if key not in _cachedImplicitVessel:
        _cachedImplicitVessel[key] = _build_vessel_implicit(segNode)
    return _cachedImplicitVessel[key]
    
def _build_vessel_implicit(segNode):
    """
    Belirlenen damar segmentlerini (VESSEL_NAMES) kapalı yüzeyden temizleyip
    tek bir vtkImplicitBoolean ('union') fonksiyonu döndürür.

    - Her sahne güncellendiğinde yeniden hesaplanır (mtime tabanlı cache)
    - Self-intersection temizliği (vtkCleanPolyData) yapılır
    - VTK 9.0'da sadece SetInput, 9.2+'da SetInputData kullanır
    - Eklenen alt objeler thread-local listede tutulur → GC-korumalı
    """
    seg   = segNode.GetSegmentation()
    mtime = segNode.GetMTime()          # sahne değişti mi?

    # Önbellek: aynı mtime ile daha önce üretildiyse tekrar hesaplama
    if getattr(_thread_local, "mtime", None) == mtime:
        return _thread_local.obj   
        
    # ---------- yeni union hazırlanıyor ----------
    from vtk import (vtkImplicitBoolean, vtkImplicitPolyDataDistance,
                     vtkCleanPolyData)
    union = vtkImplicitBoolean()
    union.SetOperationTypeToUnion()
    keep  = []                          # C++ nesnelerini GC'den koru
    added = 0   



    for name in VESSEL_NAMES:
        sid = seg.GetSegmentIdBySegmentName(name)
        if not sid:
            continue

        closed = _getClosed(segNode, sid)             # ön-cached
        if closed is None or closed.GetNumberOfPoints() == 0:
            continue

        # Self-intersection / duplicate noktalar temizliği
        clean = vtkCleanPolyData()
        clean.SetInputData(closed)
        clean.Update()
        polyC = clean.GetOutput()
        if polyC.GetNumberOfPoints() == 0:
            continue

        imp = vtkImplicitPolyDataDistance()
        # VTK sürümüne göre uygun API
        if hasattr(imp, "SetInputData"):
            imp.SetInputData(polyC)
        else:
            imp.SetInput(polyC)

        union.AddFunction(imp)
        keep.append(imp)
        added += 1

    # ------------------ cache'e kaydet ------------------
    _thread_local.obj   = union if added else None    # hiç damar yoksa None
    _thread_local.keep  = keep                       # GC tutucu
    _thread_local.mtime = mtime
    return _thread_local.obj
                
# Add TotalSegmentator module path
# Ensure TotalSegmentator extension is installed in 3D Slicer
# Alert the user if not properly configured
totalsegmentator_path = r"C:\Users\comu\AppData\Local\slicer.org\Slicer 5.9.0-2025-03-03\slicer.org\Extensions-33529\TotalSegmentator\lib\Slicer-5.9\qt-scripted-modules"
if totalsegmentator_path not in sys.path:
    sys.path.append(totalsegmentator_path)
try:
    # TotalSegmentator modülünden python_api'yi içe aktar
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.libs import download_pretrained_weights
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
    from totalsegmentator.config import has_valid_license_offline
    from totalsegmentator.config import set_license_number
    set_license_number("aca_9RQQA0E77R7XQZ")
    print("[Python] TotalSegmentator module successfully loaded.")
except ImportError as e:
    error_message = f"TotalSegmentator module could not be loaded. Please check your installation. Error: {str(e)}"
    slicer.util.errorDisplay(error_message)
    raise ImportError(error_message)
VESSEL_NAMES = [
    "heart","aorta","pulmonary_vein","brachiocephalic_trunk",
    "subclavian_artery_right","subclavian_artery_left",
    "common_carotid_artery_right","common_carotid_artery_left",
    "brachiocephalic_vein_left","brachiocephalic_vein_right",
    "superior_vena_cava","inferior_vena_cava","pulmonary_artery"
]
# Import VTK and Slicer libraries
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange
from slicer import vtkMRMLScalarVolumeNode

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.expanduser("~"), "LungBiopsyTractPlanner.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def readImageData(niftiPath: str) -> "vtkImageData":
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(niftiPath)
    reader.Update()
    img = vtk.vtkImageData()
    img.DeepCopy(reader.GetOutput())   # bağımsız kopya
    return img

LUNG_SEGMENTS = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right",
    "lung_lower_lobe_right"
]

def iter_valid_lobes(segNode, referenceVolumeNode,
                     *, return_mask=False, as_ids=False):
    """
    Yields:
      (name, mask)  if return_mask
      (name, sid)   if as_ids
      name          else
    """
    seg = segNode.GetSegmentation()
    binRep = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    seg.SetConversionParameter("ReferenceImageGeometry",referenceVolumeNode.GetID())
    if not seg.ContainsRepresentation(binRep):
        segNode.CreateBinaryLabelmapRepresentation()

    for name in LUNG_SEGMENTS:
        sid = seg.GetSegmentIdBySegmentName(name)
        if not sid:
            continue
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
                   segNode, sid, referenceVolumeNode) 
        if mask is None or not np.any(mask):
            continue
        if return_mask:
            yield (name, mask)
        elif as_ids:
            yield (name, sid)
        else:
            yield name      
        
TEMP_DIR  = r"C:/Users/user/Desktop/SlicerTemp"    # tek kaynak
OUTPUT_DIR = os.path.join(TEMP_DIR, "TotalSegmentatorOutput")

#
# LungBiopsyTractPlanner
#

class LungBiopsyTractPlanner(ScriptedLoadableModule):
    """
    Main module class handling metadata and sample data registration
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LungBiopsyTractPlanner")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "IGT")]
        self.parent.dependencies = ["TotalSegmentator"]
        self.parent.contributors = ["Saner Esmer (COMU Hosp.)"]
        self.parent.helpText = _(
            "This module assists in planning lung biopsy tracts, ensuring that safe biopsy tracts "
            "are calculated based on anatomical segmentations and user-defined constraints."
        )
        self.parent.acknowledgementText = _(
            "This module was developed by Saner Esmer, with help from the Slicer community."
        )
        slicer.app.connect("startupCompleted()", LungBiopsyTractPlanner.registerSampleData)

    @staticmethod
    def registerSampleData():
        """Register sample data for demonstration (optional)."""
        import SampleData
        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            category="LungBiopsyTractPlanner",
            sampleName="LungBiopsyTractPlanner1",
            thumbnailFileName=os.path.join(iconsPath, "LungBiopsyTractPlanner1.png"),
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            fileNames="LungBiopsyTractPlanner1.nrrd",
            checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            nodeNames="LungBiopsyTractPlanner1"
        )

@parameterNodeWrapper
class LungBiopsyTractPlannerParameterNode:
    """
    Defines parameters that can be stored in the scene and synchronized with the GUI.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode

class LungBiopsyTractPlannerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    GUI class: manages the user interface, button clicks, etc.
    Corresponds to the .ui file you shared, which has:
      - inputSelector, outputSelector, invertedOutputSelector
      - imageThresholdSliderWidget, invertOutputCheckBox
      - applyButton
      - paintButton
      - analyzeTractsButton
    """
        
    def getValidInputVolumeNode(self):
        """
        Returns the currently selected input volume node from the active Red slice.
        If no valid volume node is found, raises an exception.
        """
        # Aktif Red dilimindeki hacim düğümünü alın
        volumeNode = self.ui.inputSelector.currentNode()  # Use the inputSelector from the UI
        if not volumeNode or not volumeNode.GetImageData():
            raise ValueError("No valid input volume selected or it has no image data.")
        self.logCallback(f"[Python] Selected inputVolumeNode: {volumeNode.GetName()} ({volumeNode.GetClassName()})")
        return volumeNode
        
        
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        
        
        
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.segmentationNode = None  # We'll store user's "BiopsyRegion" in here
        self.logCallback = self.updateStatus
        
    def setup(self):
        """
        UI ilk yüklendiğinde çağrılır: .ui dosyasını açar, MRML sahnesine bağlanır
        ve sinyalleri kurar.
        """
        # ------------------------------------------------------------------
        # 0) Slicer alt-yapılarını başlat
        # ------------------------------------------------------------------
        ScriptedLoadableModuleWidget.setup(self)
        self.logCallback = self.updateStatus        # tek hat üzerinden log

        # Logic nesnesi
        self.logic = LungBiopsyTractPlannerLogic(widget=self,
                                                 logCallback=self.logCallback)
        self.initializeParameterNode()

        # ------------------------------------------------------------------
        # 1) .ui dosyasını yükle
        # ------------------------------------------------------------------
        uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/LungBiopsyTractPlanner.ui"))
        if not uiWidget:
            raise RuntimeError("UI dosyası bulunamadı / açılamadı.")

        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # **statusLog garantisi**
        assert self.ui.statusLog is not None, (
            "Qt Designer’da QTextEdit'in objectName'ini 'statusLog' yapmayı unuttun!")
        self.ui.statusLog.setReadOnly(False)   # yazılabilir kalsın

        # ------------------------------------------------------------------
        # 2) MRML sahnesi ile eşleştirmeler
        # ------------------------------------------------------------------
        uiWidget.setMRMLScene(slicer.mrmlScene)
        if hasattr(self.ui, "inputSelector"):          # Selector sahneyi görsün
            self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)

        # Sahnedeki ilk hacmi otomatik seç
        volNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if volNode:
            self._parameterNode.SetParameter("inputVolume", volNode.GetID())

        # Yeni eklenen hacimleri dinle
        self.addObserver(slicer.mrmlScene,
                         slicer.vtkMRMLScene.NodeAddedEvent,
                         self.onNodeAdded)

        # ------------------------------------------------------------------
        # 3) UI-sin yalleri bağla
        # ------------------------------------------------------------------
        self.setupConnections()

        # ------------------------------------------------------------------
        # 4) Arka-plan log kuyruğunu 500 ms’de bir temizle (BLOCK YOK)
        # ------------------------------------------------------------------
        self.pollTimer = qt.QTimer()
        self.pollTimer.setInterval(500)
        self.pollTimer.timeout.connect(self.pollLogs)
        self.pollTimer.start()

        # İlk durum mesajı
        self.updateStatus("✅ LungBiopsyTractPlanner hazır.")

    def updateStatus(self, message: str):
        """
        3D Slicer UI'deki statusLabel ve statusLog'u günceller.
        Artık sadece ana thread (GUI) içinde çağrıldığını varsayıyoruz;
        bu nedenle ek denetimlere ya da singleShot timer'a gerek kalmıyor.
        """
        # Eğer arayüz daha yüklenmemişse (ör. module reloaded'dan hemen sonra) fallback:
        
        if not self.ui or not self.ui.statusLog:
            print(f"[STATUS - fallback only] {message}")
            return

        # 1) Log ekranına satırı ekle
        self.ui.statusLog.append(message)
        self.ui.statusLog.ensureCursorVisible()

        # 2) (Opsiyonel) statusLabel'a da yazmak isterseniz
        if hasattr(self.ui, "statusLabel") and self.ui.statusLabel:
            self.ui.statusLabel.setText(message)

        # 3) Konsola bas
        print("[STATUS]", message)
        # 4) (Opsiyonel) slicer.app.processEvents() anlık yenileme gerekirse
        
        
    def pollLogs(self):
        """Ana thread içerisinde düzenli aralıklarla çağrılıp 
        arka plan outputQueue'sundaki satırları StatusLog'a basar."""
        if not self.logic:
            return
        # Kuyruğu boşalıncaya kadar okuyoruz
        while not self.logic.outputQueue.empty():
            try:
                line = self.logic.outputQueue.get_nowait()
            except Exception:
                break
            # Bu satırı statusLog'a ekle
            self.updateStatus(line)
        # logic.outputQueue.empty() olduktan sonra fonksiyon biter
        
    def setupConnections(self):
        """Connects UI buttons and signals to corresponding functions."""
        
        self.ui.paintButton.clicked.connect(self.onPaintButtonClicked)
        self.metastasisCheckBox = self.ui.metastasisCheckBox
        self.ui.segmentButton.clicked.connect(self.onSegmentButtonClicked)
        self.ui.analyzeTractsButton.clicked.connect(self.onAnalyzeTractsClicked)
      
        
    def onNodeAdded(self, caller=None, event=None, callData=None):
        node = callData  # Slicer≥5.4’te doğrudan eklenen node gelir
        if not node or not node.IsA("vtkMRMLScalarVolumeNode"):
            return

        # *** Geçici adları atla  ***
        if node.GetName().startswith("__temp__") or node.GetHideFromEditors():
            return

        # CT değilse (LabelMap/Segmentation) yine atla
        if node.GetClassName() == "vtkMRMLLabelMapVolumeNode":
            return

        self._parameterNode.SetParameter("inputVolume", node.GetID())
        self.updateStatus(f"[INFO] Input volume set to: {node.GetName()}")

    def onPaintButtonClicked(self):
        """
        Let the user paint the biopsy region in Segment Editor.
        """
        try:
            # 1. Giriş hacmini kontrol et
            inputVolumeNode = self.ui.inputSelector.currentNode()  # UI'den hacmi al

            if not inputVolumeNode:
                slicer.util.errorDisplay("No input volume selected. Please select an input volume.")
                return

            # 2. Segmentation node'u kontrol et ve oluştur
            if not hasattr(self, "segmentationNode") or not self.segmentationNode:
                self.segmentationNode = slicer.mrmlScene.GetFirstNodeByName("BiopsyRegionNode")
                
            if not self.segmentationNode:
                self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "BiopsyRegionNode")
                self.segmentationNode.CreateDefaultDisplayNodes()  # Eksik display node'ları oluştur
                slicer.util.delayDisplay("Segmentation node created. Please use the Segment Editor to paint the target area.")


            # 5. Yeni bir segment oluştur ve etkinleştir
            segmentID = "TargetRegion"
            segmentation = self.segmentationNode.GetSegmentation()
            if not segmentation.GetSegment(segmentID):
                # “AddEmptySegment” ikinci parametreyi görünür isim olarak kullanır
                # Ayrıca segmentin internal ID’si de “TargetRegion” olacak
                newSegmentID = segmentation.AddEmptySegment(segmentID, segmentID)

                # Bu satırların hemen ardından, segmentin SetName ile de tekrar garantileyebilirsiniz
                segment = segmentation.GetSegment(segmentID)
                segment.SetName(segmentID)
           
            # Segmentin rengini ayarla
            self.segmentationNode.GetSegmentation().GetSegment(segmentID).SetColor(1.0, 1.0, 0.0)  # Sarı renk
                
            # 6. Segmentasyon düğümünün görselleştirme ayarlarını yap
            displayNode = self.segmentationNode.GetDisplayNode()
            if displayNode:
                # Genel 2D görünürlük
                displayNode.SetVisibility2DFill(True)  # Tüm segmentler için 2D dolgu görünürlüğü
                displayNode.SetVisibility2DOutline(True)  # Tüm segmentler için 2D dış hat görünürlüğü

                # TargetRegion için özel 2D ve 3D saydamlık ayarları
                displayNode.SetSegmentOpacity2DFill(segmentID, 0.5)  # 2D dolgu saydamlığı
                displayNode.SetSegmentOpacity3D(segmentID, 0.3)  # 3D saydamlık

            # 7. Segment Editor'e geçiş yap (En son adımda yapılır)
            slicer.util.selectModule("SegmentEditor")


        except Exception as e:
            slicer.util.errorDisplay(f"Error in onPaintButtonClicked: {e.__class__.__name__}: {str(e)}")
        
    def onSegmentButtonClicked(self):
        """
        Sadece TotalSegmentator segmentasyonunu çalıştırır.
        Kullanıcı segmentasyon bittikten sonra 'Trakt Analizi'ne geçebilir.
        """
        try:
            inputVolumeNode = self.ui.inputSelector.currentNode()
            if not inputVolumeNode:
                slicer.util.errorDisplay("Önce CT hacmini seçmelisiniz.")
                return

            # Varolan segmentasyon düğümü varsa yeniden çalıştırmak isteyip istemediğini sor
            existing = slicer.util.getNode(pattern="^CombinedSegmentation$") if slicer.mrmlScene.GetFirstNodeByName("CombinedSegmentation") else None
            if existing and not slicer.util.confirmYesNoDisplay(
                    "Segmentasyon zaten var. Yeniden çalıştırmak uzun sürebilir.\nYine de devam edilsin mi?"):
                return
            if existing:
                slicer.mrmlScene.RemoveNode(existing)

            self.updateStatus("[STATUS] TotalSegmentator segmentasyonu başlatılıyor…")
            combinedSegmentationNode = self.logic.runTotalSegmentatorSequentially(inputVolumeNode)
            if not combinedSegmentationNode:
                slicer.util.errorDisplay("Segmentasyon başarısız oldu.")
                return

            self.updateStatus("[STATUS] Segmentasyon tamamlandı.")

        except Exception as e:
            slicer.util.errorDisplay(f"Segmentasyon sırasında hata: {str(e)}")
        
            
    def onAnalyzeTractsClicked(self):
        """
        Kullanıcının seçtiği CT + mevcut segmentasyon ile
        biyopsi traktlarını hesaplar ve sahneye çizer.
        """
        self.logCallback("[DEBUG] onAnalyzeTractsClicked() başladı…")

        try:
            # ------------------------------------------------------------------
            # 1) Giriş hacmi kontrolü
            # ------------------------------------------------------------------
            inputVolumeNode = self.ui.inputSelector.currentNode()
            if not inputVolumeNode:
                slicer.util.errorDisplay("Önce bir CT hacmi seçmelisiniz.")
                return
            self.logCallback(f"[DEBUG] Seçilen giriş hacmi: {inputVolumeNode.GetName()}")

            # ------------------------------------------------------------------
            # 2) Segmentasyon düğümünü esnek biçimde yakala
            #    (CombinedSegmentation → TotalSegmentation → ilk segmentation)
            # ------------------------------------------------------------------
            segNode = (
                slicer.mrmlScene.GetFirstNodeByName("CombinedSegmentation")
                or slicer.mrmlScene.GetFirstNodeByName("TotalSegmentation")
                or slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            )
            if not segNode:
                slicer.util.errorDisplay(
                    "Segmentasyon bulunamadı.\n"
                    "Lütfen önce 'Segment' düğmesine basarak TotalSegmentator’ü çalıştırın."
                )
                return
            self.logCallback(f"[INFO] Segmentasyon düğümü bulundu: {segNode.GetName()}")

            # ------------------------------------------------------------------
            # 3) Trakt analizi
            # ------------------------------------------------------------------
            self.logCallback("[INFO] Trakt analizi başlatılıyor…")
            self.logic.analyzeAndVisualizeTracts(segNode, segNode, {})
            slicer.util.delayDisplay("Trakt analizi başarıyla tamamlandı.")
            self.logCallback("[DEBUG] onAnalyzeTractsClicked() başarıyla tamamlandı.")

        except Exception as e:
            slicer.util.errorDisplay(f"Trakt analizi sırasında hata: {e}")
            traceback.print_exc()


    def cleanup(self) -> None:
        self.removeObservers()

    def initializeParameterNode(self):
        """
        Create or retrieve the parameter node for the module.
        """
        # Logic sınıfından parametre düğümünü al
        if not hasattr(self, "_parameterNode") or self._parameterNode is None:
            self._parameterNode = self.logic.getParameterNode()

        # Eğer hala boşsa, kullanıcıya hata mesajı göster
        if not self._parameterNode:
            slicer.util.errorDisplay("Failed to initialize parameter node.")

    def setParameterNode(self, inputParameterNode: Optional[LungBiopsyTractPlannerParameterNode]) -> None:
        """
        Set the parameter node and observe it. Updates the GUI to reflect the parameter node state.
        """
        # Eski düğümü temizle
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode

        # Yeni düğümü bağla ve UI ile senkronize et
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            self.updateGUIFromParameterNode()

        
    def updateGUIFromParameterNode(self):
        """
        Update the GUI to reflect the current state of the parameter node.
        """
        if not self._parameterNode or not self.ui:
            return




#
# LungBiopsyTractPlannerLogic
#



class LungBiopsyTractPlannerLogic(ScriptedLoadableModuleLogic):
    """
    Implements all the core computations, constraints, and risk calculations.
    """


    def crop_volume_superoinferior(self,
        *,
        input_volume_node: slicer.vtkMRMLScalarVolumeNode,
        needle_len_max_mm: float,
        target_node_name: str = "BiopsyRegionNode",
        target_segment_name: str = "TargetRegion",
        ) -> slicer.vtkMRMLScalarVolumeNode:
        """
        TargetRegion’un üstünde/altında Δ’dan (≈ iğne diyagonal uzunluğu)
        fazla dilim varsa fazlalığı kırpar; tampon dilim sayısı yetersizse
        hacme dokunmaz ve orijinali döndürür.
        """
        # 1) TargetRegion mask
        tgt_node = slicer.mrmlScene.GetFirstNodeByName(target_node_name)
        if (not tgt_node) or (not tgt_node.IsA("vtkMRMLSegmentationNode")):
            self.logCallback(f"[WARN] Node '{target_node_name}' bulunamadı → crop iptal.")
            return input_volume_node  
        seg = tgt_node.GetSegmentation()
        tid = seg.GetSegmentIdBySegmentName(target_segment_name)
        if not tid:
            self.logCallback(f"[WARN] Segment '{target_segment_name}' bulunamadı → crop iptal.")
            return input_volume_node

        seg.SetConversionParameter("ReferenceImageGeometry",
                                   input_volume_node.GetID())
        binName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
        if not seg.ContainsRepresentation(binName):
            tgt_node.CreateBinaryLabelmapRepresentation()

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            tgt_node, tid, input_volume_node)
        if mask is None or not np.any(mask):
            raise RuntimeError("TargetRegion mask’i boş!")

        # 2) Dilim sınırları & Δ
        z_inds = np.where(mask.any(axis=(1, 2)))[0]
        z_sup, z_inf = int(z_inds[0]), int(z_inds[-1])
        n_slices     = mask.shape[0]
        dz           = input_volume_node.GetSpacing()[2]                # mm
        delta_raw    = int(round(needle_len_max_mm / (math.sqrt(2)*dz)))

        sup_room = z_sup                    # Üst boşluk
        inf_room = n_slices - 1 - z_inf     # Alt boşluk

        # Üst-alt tampon yetersizse: hiç kırpma
        if sup_room <= delta_raw and inf_room <= delta_raw:
            return input_volume_node

        lo = 0 if sup_room <= delta_raw else z_sup - delta_raw
        hi = n_slices - 1 if inf_room <= delta_raw else z_inf + delta_raw

        # 3) ROI & CropVolume
        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roi.CreateDefaultDisplayNodes()
        M = vtk.vtkMatrix4x4()
        input_volume_node.GetIJKToRASMatrix(M)
        ijk_min, ijk_max = [0, 0, lo, 1], [mask.shape[2]-1,
                                           mask.shape[1]-1, hi, 1]
        ras_min, ras_max = [0, 0, 0, 1], [0, 0, 0, 1]
        M.MultiplyPoint(ijk_min, ras_min)
        M.MultiplyPoint(ijk_max, ras_max)
        center = [(ras_min[i] + ras_max[i]) * 0.5 for i in range(3)]
        size   = [max(abs(ras_max[i] - ras_min[i]), 1.0) for i in range(3)]

        roi.SetCenter(*center)
        roi.SetSize(*size)

        p = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropped = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        cropped.SetName(f"{input_volume_node.GetName()}_cropped")
        p.SetInputVolumeNodeID(input_volume_node.GetID())
        p.SetROINodeID(roi.GetID())
        p.SetOutputVolumeNodeID(cropped.GetID())
        p.SetVoxelBased(True)
        p.SetIsotropicResampling(False)

        slicer.modules.cropvolume.logic().Apply(p)

        # Temizlik
        slicer.mrmlScene.RemoveNode(roi)
        slicer.mrmlScene.RemoveNode(p)

        return cropped
   
    def renameSegmentsFromHeader(self, segmentationNode, mlNiftiPath):
        """
        TotalSegmentator multilabel NIfTI'sinin extended header’ındaki
        label sözlüğünü okuyup sahnedeki Segment_XX adlarını gerçek
        anatomi isimleriyle değiştirir.
        """
        _, labelDict = load_multilabel_nifti(mlNiftiPath)  # {value: "aorta", ...}

        seg = segmentationNode.GetSegmentation()
        for segId in seg.GetSegmentIDs():
            oldName = seg.GetSegment(segId).GetName()

            # Varsayılan ad "Segment_23" ise etiketi ayıkla
            if not oldName.startswith("Segment_"):
                continue
            try:
                labelValue = int(oldName.split("_")[-1])
            except ValueError:
                continue

            newName = labelDict.get(labelValue)
            if newName:
                seg.GetSegment(segId).SetName(newName)
                
    def getSegmentMaskAsArray(self, segNode, segId, refVol):
        segLogic = slicer.modules.segmentations.logic()
        tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        segLogic.ExportSegmentsToLabelmapNode(segNode, [segId], tmp, refVol)
        arr = slicer.util.arrayFromVolume(tmp).astype(np.uint8, copy=False)
        slicer.mrmlScene.RemoveNode(tmp)
        return arr
    
    def segmentHasContent(self, segNode, segmentName):
        """
        segNode      : vtkMRMLSegmentationNode (örn. combinedSegmentationNode)
        segmentName  : aranan segmentin görünen ismi (örn. 'heart')
        Dönüş        : True → segment var **ve** en az 1 voxel içeriyor
                       False → segment yok **veya** tamamen boş
        """
        seg = segNode.GetSegmentation()
        segId = seg.GetSegmentIdBySegmentName(segmentName)
        if not segId:
            return False  # Segment hiç yok

        # Segmentin voxel sayısını hesapla
        arr = self.getSegmentMaskAsArray(segNode, segId, self.inputVolumeNode)
        return np.any(arr)         # True → ≥1 voxel var, False → hepsi 0
            
                
    def getSmoothedClosedSurfaceRepresentation(self, segmentationNode, segmentName, iterations=50, relaxation=0.3, boundarySmoothingOn=True):
        """
        Verilen segmentationNode içindeki 'segmentName' isimli segmentin kapalı yüzey temsilini alır,
        smoothing filtresi uygulayarak yumuşatır ve segmenti silip yerine yeni bir segment
        (smoothed polydata) ekler. En son, smoothed polydata'yı döndürür.

        Parametreler:
          - segmentationNode: vtkMRMLSegmentationNode (segmentasyon düğümü)
          - segmentName: İşlem yapılacak segmentin adı (ör. "lung_upper_lobe_left")
          - iterations: Smoothing filtresinin iterasyon sayısı (varsayılan 50)
          - relaxation: Relaxation factor (varsayılan 0.3)
          - boundarySmoothingOn: Sınır boyunca smoothing uygulanır (varsayılan True)

        Döndürür:
          - smoothedPolyData: vtkPolyData, yumuşatılmış kapalı yüzey temsili.
        """
        segmentation = segmentationNode.GetSegmentation()
        segmentID = segmentation.GetSegmentIdBySegmentName(segmentName)
        if not segmentID:
            print(f"[ERROR] Segment '{segmentName}' not found.")
            return None

        repName = "ClosedSurface"
        # Varsa kapalı yüzey temsili, yoksa oluştur
        if not segmentation.ContainsRepresentation(repName):
            segmentation.CreateRepresentation(repName)

        # Mevcut kapalı yüzey polydata'sını al
        currentPolyData = vtk.vtkPolyData()
        slicer.modules.segmentations.logic().GetSegmentClosedSurfaceRepresentation(segmentationNode, segmentID, currentPolyData)
        if currentPolyData.GetNumberOfPoints() == 0:
            print(f"[ERROR] Segment '{segmentName}' closed surface could not be retrieved (no points).")
            return None

        # PolyData smoothing filtresi
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(currentPolyData)
        smoothFilter.SetNumberOfIterations(iterations)
        smoothFilter.SetRelaxationFactor(relaxation)
        smoothFilter.FeatureEdgeSmoothingOn()  # Kenarları korumamak için
        if boundarySmoothingOn:
            smoothFilter.BoundarySmoothingOn()
        else:
            smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()

        smoothedPolyData = smoothFilter.GetOutput()
        numPts = smoothedPolyData.GetNumberOfPoints()
        if numPts == 0:
            print("[ERROR] Smoothing failed. No points in output.")
            return None

        # Orijinal segmentin renk, isim ve tag bilgilerini sakla
        originalSegment = segmentation.GetSegment(segmentID)
        origColor = originalSegment.GetColor()
        origName = originalSegment.GetName()
        
        # Eski segmenti kaldır
        segmentation.RemoveSegment(segmentID)
        
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Temp_SmoothedSurface")
        modelNode.SetAndObservePolyData(smoothedPolyData)
        
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(modelNode) 
        
        allSegmentIDs = segmentation.GetSegmentIDs()
        if not allSegmentIDs:
            print("[ERROR] No segments found after import!")
            return smoothedPolyData

        # Son eklenen segment, listenin/tuple’ın en son elemanıdır
        newSegmentID = allSegmentIDs[-1]
        

        # Yeni segmente eski renk, isim ve tag'leri geri yükle
        newSeg = segmentation.GetSegment(newSegmentID)
        if newSeg:
            newSeg.SetColor(origColor)
            newSeg.SetName(origName)

        print(f"[INFO] '{segmentName}' segmenti için smoothing uygulandı. Nokta sayısı: {numPts}")
        return smoothedPolyData

    def runTaskSequentially(self, inputVolumeNode, task, fast, device="cpu", excludeSegments=None):
        """
        Belirtilen task için TotalSegmentator komutunu çalıştırır, tamamlanmasını bekler 
        ve çıkan segmentation node’u döndürür.
        """
        #defaultColorMap = LungBiopsyTractPlannerLogic.getDefaultColorMapping()
        
        # Geçici dizin, input ve output yollarını ayarla
        temp_dir = "C:/Users/user/Desktop/SlicerTemp"
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "TotalSegmentatorOutput")
        if task == "total":
            if os.path.exists(output_path):
                import shutil
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
        else:
            # Diğer görevler için; klasör varsa olduğu gibi kullanılır,
            # yoksa oluşturulur.
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
  
        input_path = os.path.join(temp_dir, "input_image.nii.gz")
        input_path = os.path.normpath(input_path)
        
        self.widget.updateStatus(f"[INFO] {task} segmentasyonu başlatılıyor...")
 
        # Input volume’u kaydet
        save_result = slicer.util.saveNode(inputVolumeNode, input_path)
        
        if not save_result or not os.path.exists(input_path):
            self.logCallback(f"[ERROR] ({task}) Input volume {input_path} kaydedilemedi.")
            raise IOError(f"({task}) Input volume {input_path} kaydedilemedi.")

        self.logCallback(f"[INFO] {task} segmentasyonu başlatılıyor...")
        
        if task == "total":
            roi_subset_included = [
                "spleen", "liver", "stomach", "esophagus", "thyroid_gland", "brachiocephalic_vein_left", "brachiocephalic_vein_right",
                "heart", "atrial_appendage_left", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "superior_vena_cava", "inferior_vena_cava",
                "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right", "common_carotid_artery_left",
                "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right","lung_lower_lobe_right",
                "autochthon_left", "autochthon_right", "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
                "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
                "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum", "costal_cartilages",
                "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", 
                "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "humerus_left", "humerus_right"]

            roi_subset_str = repr(roi_subset_included)

            # Çalıştırılacak komutu oluşturun:
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                (
                    f"from totalsegmentator.python_api import totalsegmentator;"
                    f"totalsegmentator(input=r'{input_path}', "
                    f"output=r'{output_path}', "
                    f"task='{task}', fastest=True, device='cpu', "
                    f"roi_subset={roi_subset_str}, "
                    "ml=True)" 
                )                    
            ]    
        elif task == "tissue_types":
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            license_number = "aca_9RQQA0E77R7XQZ" 
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=False, device='cpu', license_number='{license_number}', "
                "ml=True)" 
            ]
        elif task == "lung_vessels":
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            license_number = "aca_9RQQA0E77R7XQZ"
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=False, device='cpu', license_number='{license_number}', "
                "ml=True)" 
            ]
        elif task == "body":
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            license_number = "aca_9RQQA0E77R7XQZ"
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=True, device='cpu', license_number='{license_number}', "
                "ml=True)" 
            ]
        elif task == "pleural_pericard_effusion":
            license_number = "aca_9RQQA0E77R7XQZ" 
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=False, device='cpu', license_number='{license_number}', "
                "ml=True)" 
            ]                    
        elif task == "heartchambers_highres":
            license_number = "aca_9RQQA0E77R7XQZ" 
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=False, device='cpu', license_number='{license_number}', "
                "ml=True)"             
            ]   
        elif task == "lung_nodules":
            license_number = "aca_9RQQA0E77R7XQZ" 
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{task}_Segmentation")
            command = [
                "C:/Users/comu/AppData/Local/slicer.org/Slicer 5.9.0-2025-03-03/bin/PythonSlicer.exe",
                "-c",
                f"from totalsegmentator.python_api import totalsegmentator;"
                f"totalsegmentator(input=r'{input_path}', output=r'{output_path}', task='{task}', fast=False, device='cpu', license_number='{license_number}', "
                "ml=True)"             
            ]    
        self.logCallback(f"[DEBUG] ({task}) Çalıştırılacak komut: {command}")
        # ✅ **Subprocess ile başlat**
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False
            )
        except Exception as e:
            self.logCallback(f"[ERROR] ({task}) TotalSegmentator başlatılamadı: {str(e)}")
            slicer.util.errorDisplay(f"TotalSegmentator çalıştırılamadı:\n{e}\n\nİpuçları:\n • TOTALSEG_FORCE_CPU=1 (CPU)\n • TOTALSEG_FAST=1 (low‑mem)\n • CT’yi Crop Volume ile küçült")
            return None            
        # ✅ **Canlı olarak logları oku ve StatusLog’a yaz**
        def read_output(stream, log_type):
            while True:
                raw_line = stream.readline()  # ham bayt okuma
                if raw_line == b"" and process.poll() is not None:
                    break
                if raw_line:
                    # UTF-8 decode ederken hatalı karakterleri '?' ile değiştir
                    decoded_line = raw_line.decode("utf-8", errors="replace")
                    text = f"[{log_type}] {decoded_line.strip()}"
                    self.outputQueue.put(text)
                    self.logCallback(text)
                    slicer.app.processEvents()  # UI'nin donmasını engelle
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "INFO"))
        stdout_thread.daemon = True
        stdout_thread.start()
        #stdout_thread.join()
        return_code = process.wait()
        if return_code == 0:
            self.logCallback("[SUCCESS] İşlem başarıyla tamamlandı. ✅")
        else:
            self.logCallback("[ERROR] İşlem başarısız oldu! ❌")

        # --- Çok‑label dosyasını nerede olursa olsun ara ---
        candidates = glob.glob(os.path.join(output_path, "segmentations.nii.gz")) + \
                     glob.glob(os.path.join(output_path, "segmentations", "segmentations.nii.gz")) + \
                     glob.glob(os.path.join(os.path.dirname(output_path), "TotalSegmentatorOutput.nii")) + \
                     glob.glob(os.path.join(os.path.dirname(output_path), "TotalSegmentatorOutput.nii.gz"))
        ml_path = candidates[0] if candidates else None
        if ml_path and os.path.exists(ml_path):
            combinedNode = slicer.util.loadSegmentation(ml_path)
            combinedNode.SetName("CombinedSegmentation")
            combinedNode.GetSegmentation().SetConversionParameter("ReferenceImageGeometry", inputVolumeNode.GetID())
            self.renameSegmentsFromHeader(combinedNode, ml_path)
            self.logCallback(f"[SUCCESS] Multilabel segmentasyon yüklendi: {os.path.basename(ml_path)} ✅")
            return combinedNode
        else:
            self.logCallback("[ERROR] Multilabel dosya gerçekten bulunamadı!")  
        return None
 
    def addTargetRegionFromLargestNodule(self, combinedSegmentationNode, inputVolumeNode):
        seg = combinedSegmentationNode.GetSegmentation()
        lungNodulesID = seg.GetSegmentIdBySegmentName("lung_nodules")
        if not lungNodulesID:
            self.logCallback("[UYARI] 'lung_nodules' segmenti bulunamadı.")
            return

        mask = slicer.util.arrayFromSegmentBinaryLabelmap(combinedSegmentationNode, lungNodulesID, inputVolumeNode)
        import scipy.ndimage as ndi
        labeled, num_features = ndi.label(mask)
        if num_features == 0:
            self.logCallback("[UYARI] 'lung_nodules' içinde hiç nodül bulunamadı.")
            return

        sizes = ndi.sum(mask, labeled, index=range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        largest_mask = (labeled == largest_label).astype(np.uint8)

        tempLabelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "__tempTargetRegion__")
        slicer.util.updateVolumeFromArray(tempLabelNode, largest_mask)
        tempLabelNode.SetSpacing(inputVolumeNode.GetSpacing())
        tempLabelNode.SetOrigin(inputVolumeNode.GetOrigin())
        matrix = vtk.vtkMatrix4x4()
        inputVolumeNode.GetIJKToRASMatrix(matrix)
        tempLabelNode.SetIJKToRASMatrix(matrix)

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tempLabelNode, combinedSegmentationNode)
        seg = combinedSegmentationNode.GetSegmentation()
        lastID = seg.GetNthSegmentID(seg.GetNumberOfSegments() - 1)
        seg.GetSegment(lastID).SetName("TargetRegion")
        seg.GetSegment(lastID).SetColor(1.0, 1.0, 0.0)

        slicer.mrmlScene.RemoveNode(tempLabelNode)
        self.logCallback("[INFO] TargetRegion otomatik olarak en büyük nodülden oluşturuldu.")
    
    def runTotalSegmentatorSequentially(self, inputVolumeNode):
        """
        Tüm TotalSegmentator görevlerini sırasıyla çalıştırır.
        - İlk olarak "total" görevi çalıştırılır; çıktısı birleşik segmentation node (CombinedSegmentation) olarak kabul edilir.
        - Sonraki görevler ("lung_vessels", "pleural_pericard_effusion", "body") çalıştırılır, ve çıkan segmentler, 
          eğer birleşik node’da aynı isimde segment yoksa, eklenir; ekleme bittikten sonra o görevin node’u silinir.
        - Ardından Emphysema/Bulla segmentasyonu çalıştırılıp, aynı şekilde birleşik node’ya eklenir, 
          ve node silinir.
        - Son olarak, kullanıcının boyduğu "TargetRegion" segmenti de birleşik node’ya eklenir ve 
          BiopsyRegionNode silinir.
        - Son olarak, birleşik node subject hierarchy’de input volume’un altında konumlandırılır.
        """
        
        temp_dir = r"C:\Users\user\Desktop\SlicerTemp" 
        
        # 1. İlk "total" görevi: ROI subset dikkate alınarak çalıştırılır.
        self.widget.updateStatus("[INFO] Tüm segmentasyon işlemleri başlatılıyor...")
        
        originalVolumeNode = inputVolumeNode  # yedek

        try:
            inputVolumeNode = self.crop_volume_superoinferior(
                input_volume_node=originalVolumeNode,
                needle_len_max_mm=190.0      # ← kliniğine göre değiştir
            )
        except RuntimeError as e:
            self.logCallback(f"[WARN] Cropping skipped: {e}")
            inputVolumeNode = originalVolumeNode

        
        # TotalSegmentator burada croppedVolume ile çağrılacak
        totalNode = self.runTaskSequentially(inputVolumeNode, "total", fast=True, device="cpu", excludeSegments=None)
        if totalNode is None:
            self.logCallback("[ERROR] total segmentasyonu başarısız oldu!")
            return None
            
        segmentation = totalNode.GetSegmentation()
        segmentation.SetConversionParameter("ReferenceImageGeometry", inputVolumeNode.GetID())
        self.logCallback(f"[INFO] ReferenceImageGeometry parametresi ayarlandı: {inputVolumeNode.GetID()}")
            
        totalNode.SetName("CombinedSegmentation")
        combinedSegmentationNode = totalNode


        # --- Yeni Ekleme: Akciğer lob segmentasyonları için smoothing uygulaması ---

        for segmentName in LUNG_SEGMENTS:
            smoothedPoly = self.getSmoothedClosedSurfaceRepresentation(
                combinedSegmentationNode, 
                segmentName, 
                iterations=50, 
                relaxation=0.3, 
                boundarySmoothingOn=True
            )
            if smoothedPoly:
                self.logCallback(f"[INFO] {segmentName} segmenti için smoothing uygulandı. Nokta sayısı: {smoothedPoly.GetNumberOfPoints()}")
            else:
                self.logCallback(f"[WARNING] {segmentName} segmenti için smoothing yapılamadı.")
            
        # 2. Total task tamamlannıp akciğer smoothing yapıldıktan sonra diğer kalan görevlerin sırayla çalıştırılması 
        remainingTasks = ["pleural_pericard_effusion", "lung_nodules", "lung_vessels", "tissue_types", "body", "heartchambers_highres"]

        for task in remainingTasks:
            self.logCallback(f"[INFO] {task} segmentasyonu başlatılıyor...")
            
            taskNode = self.runTaskSequentially(inputVolumeNode, task, fast=False, device="cpu", excludeSegments=None)
            if taskNode:
                segLogic = slicer.modules.segmentations.logic()
                # Tüm segmentleri, isim bazında kontrol ederek ekleyelim
                taskSegmentation = taskNode.GetSegmentation()
                combinedSegmentation = combinedSegmentationNode.GetSegmentation()
                for i in range(taskSegmentation.GetNumberOfSegments()):
                    segmentID = taskSegmentation.GetNthSegmentID(i)
                    segment = taskSegmentation.GetSegment(segmentID)
                    segName = segment.GetName()
                    # Kontrol: Eğer birleşik node içinde aynı isimde bir segment varsa, ekleme yapmayalım.
                    exists = False
                    for existingID in combinedSegmentation.GetSegmentIDs():
                        if combinedSegmentation.GetSegment(existingID).GetName() == segName:
                            exists = True
                            break
                    if not exists:
                        # Segmenti geçici labelmap'e aktar ve ardından birleşik node'ya import et.
                        tempLabelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                        segLogic.ExportSegmentsToLabelmapNode(taskNode, [segmentID], tempLabelmap)
                        segLogic.ImportLabelmapToSegmentationNode(tempLabelmap, combinedSegmentationNode)
                        slicer.mrmlScene.RemoveNode(tempLabelmap)
                        self.logCallback(f"[DEBUG] '{segName}' segmenti {task} görevinden birleşik node'a eklendi.")
                    else:
                        self.logCallback(f"[DEBUG] '{segName}' segmenti zaten birleşik node'da; eklenmedi.")
                # Görevin tamamlanmasının ardından, artık bu taskNode'yu sahneden silelim.
                slicer.mrmlScene.RemoveNode(taskNode)
                self.logCallback(f"[INFO] {task} segmentasyonu birleşik node'a eklenip, kaynak node silindi.")

            else:
                self.logCallback(f"[WARNING] {task} segmentasyonu başarısız oldu, atlanıyor.")
       
  

        # 3. Emphysema/Bulla segmentasyonunu çalıştır ve birleşik node'a ekle
        self.logCallback("[INFO] Emphysema/Bulla segmentasyonu başlatılıyor...")
        seg_emphysema = self.createEmphysemaSegment(inputVolumeNode, combinedSegmentationNode)
        
        if seg_emphysema:
            segLogic = slicer.modules.segmentations.logic()
            emphysemaSegmentation = seg_emphysema.GetSegmentation()
            for i in range(emphysemaSegmentation.GetNumberOfSegments()):
                segmentID = emphysemaSegmentation.GetNthSegmentID(i)
                
                tempLabelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                segLogic.ExportSegmentsToLabelmapNode(seg_emphysema, [segmentID], tempLabelmap)
                segLogic.ImportLabelmapToSegmentationNode(tempLabelmap, combinedSegmentationNode)
                slicer.mrmlScene.RemoveNode(tempLabelmap)
                
                segName = emphysemaSegmentation.GetSegment(segmentID).GetName()
                self.logCallback(f"[DEBUG] '{segName}' segmenti Emphysema görevinden birleşik node'a eklendi.")
                
            slicer.mrmlScene.RemoveNode(seg_emphysema)
                
        else:
            self.logCallback("[WARNING] Emphysema/Bulla segmentasyonu başarısız oldu.")
            
            
        # 4. TargetRegion segmentini ekle
        targetNode = slicer.mrmlScene.GetFirstNodeByName("BiopsyRegionNode")
        if targetNode:
            segLogic = slicer.modules.segmentations.logic()
            targetSegmentation = targetNode.GetSegmentation()
            targetSegmentID = targetSegmentation.GetSegmentIdBySegmentName("TargetRegion")
            if targetSegmentID:
                tempLabelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                segLogic.ExportSegmentsToLabelmapNode(targetNode, [targetSegmentID], tempLabelmap)
                segLogic.ImportLabelmapToSegmentationNode(tempLabelmap, combinedSegmentationNode)
                slicer.mrmlScene.RemoveNode(tempLabelmap)
                # Sil: Kullanıcının boyduğu node artık birleşik node'a aktarıldı.
                slicer.mrmlScene.RemoveNode(targetNode)
                self.logCallback("[INFO] 'TargetRegion' segmenti birleşik node'a eklendi ve BiopsyRegionNode silindi.")
            else:
                self.logCallback("[WARNING] 'TargetRegion' segmenti bulunamadı, eklenemedi.")
                
                
        else:
            self.logCallback("[WARNING] 'BiopsyRegionNode' bulunamadı, kullanıcı bölgesi eklenemedi.Nodül tespiti başlatıldı...")
            self.addTargetRegionFromLargestNodule(combinedSegmentationNode, inputVolumeNode)
   
        # ─────────────────────────────────────────────────────────────────────────────
        # ▶ MERGE BLOĞU  –  pericardial/pleural işlemleri  +  TargetRegion düzeltmesi
        # ─────────────────────────────────────────────────────────────────────────────

        # CombinedSegmentation geometrisini mutlaka sabitle
        combinedSegmentationNode.GetSegmentation().SetConversionParameter(
            "ReferenceImageGeometry", inputVolumeNode.GetID())
        self.logCallback("[DEBUG] (Merge) ReferenceImageGeometry garanti altına alındı.")

        seg = combinedSegmentationNode.GetSegmentation()
        pulmID = seg.GetSegmentIdBySegmentName("pulmonary_vein")
        periID = seg.GetSegmentIdBySegmentName("pericardial_effusion")

        # ---------------------------------------------------------------------------
        # 1)  PERICARDIAL ➜  PLEURAL  (yalnızca komşu DEĞİLSE çalışır)
        # ---------------------------------------------------------------------------
        # ─────────────────────────────────────────────────────────────────────────────
        # 🔹 1) YARDIMCI FONKSİYONLAR – ÖNCE TANIM, SONRA KULLAN
        # ─────────────────────────────────────────────────────────────────────────────
        def _reimportPleural(maskArrayUInt8, logTag):
            if maskArrayUInt8 is None:
                self.logCallback(f"[WARNING] Pleural mask array is None — skipping reimport. ({logTag})")
                return

            maskArray = maskArrayUInt8.astype(np.uint8, copy=False)
            segLogic = slicer.modules.segmentations.logic()

            tmpVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.util.updateVolumeFromArray(tmpVol, maskArray)
            tmpVol.SetOrigin(inputVolumeNode.GetOrigin())
            tmpVol.SetSpacing(inputVolumeNode.GetSpacing())
            M = vtk.vtkMatrix4x4(); inputVolumeNode.GetIJKToRASMatrix(M)
            tmpVol.SetIJKToRASMatrix(M)

            oldID = seg.GetSegmentIdBySegmentName("pleural_effusion")
            if oldID:
                seg.RemoveSegment(oldID)

            addedIDs = vtk.vtkStringArray()
            segLogic.ImportLabelmapToSegmentationNode(tmpVol, combinedSegmentationNode, addedIDs)
            slicer.mrmlScene.RemoveNode(tmpVol)

            if addedIDs.GetNumberOfValues() == 0:
                self.logCallback(f"[ERROR] Pleural import failed ({logTag}).")
                return

            newID = addedIDs.GetValue(0)
            seg.GetSegment(newID).SetName("pleural_effusion")

            if not np.count_nonzero(maskArray):
                seg.RemoveSegment(newID)
                self.logCallback(f"[INFO] Pleural_effusion tamamen temizlendi ({logTag}).")
            else:
                combinedSegmentationNode.CreateClosedSurfaceRepresentation()
                self.logCallback(f"[INFO] Pleural_effusion güncellendi ({logTag}).")


        def _subtractTargetFromPleural():
            pleuID = seg.GetSegmentIdBySegmentName("pleural_effusion")
            tarID  = seg.GetSegmentIdBySegmentName("TargetRegion")
            if not (pleuID and tarID):
                return

            pleuArr = self.getSegmentMaskAsArray(combinedSegmentationNode, pleuID,
                                                 inputVolumeNode).astype(bool)
            tarArr  = self.getSegmentMaskAsArray(combinedSegmentationNode, tarID,
                                                 inputVolumeNode).astype(bool)

            if not np.any(pleuArr & tarArr):
                self.logCallback("[DEBUG] Pleural–TargetRegion çakışması yok.")
                return

            pleuArr[tarArr] = 0
            _reimportPleural(pleuArr.astype(np.uint8), "Target temizle")


        def _subtractLobesFromPleural():
            pleuID = seg.GetSegmentIdBySegmentName("pleural_effusion")
            if not pleuID:
                return

            pleuArr = self.getSegmentMaskAsArray(combinedSegmentationNode, pleuID,
                                                 inputVolumeNode).astype(bool)

            union = np.zeros_like(pleuArr, dtype=bool)
            for _, lobID in iter_valid_lobes(combinedSegmentationNode, inputVolumeNode, as_ids=True):
                union |= self.getSegmentMaskAsArray(combinedSegmentationNode, lobID,
                                                    inputVolumeNode).astype(bool)

            if not np.any(pleuArr & union):
                self.logCallback("[DEBUG] Pleural–Lob çakışması yok.")
                return

            updated = np.logical_and(pleuArr, np.logical_not(union))
            _reimportPleural(updated.astype(np.uint8), "Lob temizle")


        # ─────────────────────────────────────────────────────────────────────────────
        # 🔹 2) PERICARDIAL ➜ PLEURAL  (yalnız komşu DEĞİLSE)
        # ─────────────────────────────────────────────────────────────────────────────
        if periID and self.segmentHasContent(combinedSegmentationNode, "pericardial_effusion"):

            # Pulmonary-vein komşuluğu kontrolü
            adjacent = False
            if pulmID:
                segLogic = slicer.modules.segmentations.logic()
                tmpPulm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                tmpPeri = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                segLogic.ExportSegmentsToLabelmapNode(combinedSegmentationNode, [pulmID], tmpPulm, inputVolumeNode)
                segLogic.ExportSegmentsToLabelmapNode(combinedSegmentationNode, [periID], tmpPeri, inputVolumeNode)

                from scipy.ndimage import binary_dilation
                dilate_kernel = np.ones((3, 3, 3), dtype=bool)
                pulmArr = binary_dilation(slicer.util.arrayFromVolume(tmpPulm).astype(bool), structure=dilate_kernel)
                periArr = slicer.util.arrayFromVolume(tmpPeri).astype(bool)
                adjacent = np.any(pulmArr & periArr)

                slicer.mrmlScene.RemoveNode(tmpPulm)
                slicer.mrmlScene.RemoveNode(tmpPeri)

            # Komşu DEĞİLSE merge
            if not adjacent:
                self.logCallback("[INFO] Pericardial effusion → pleural_effusion aktarılıyor…")

                segLogic = slicer.modules.segmentations.logic()
                tmpPeri = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                tmpPleu = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
                segLogic.ExportSegmentsToLabelmapNode(combinedSegmentationNode, [periID], tmpPeri, inputVolumeNode)

                pleuID = seg.GetSegmentIdBySegmentName("pleural_effusion")
                if pleuID:
                    segLogic.ExportSegmentsToLabelmapNode(combinedSegmentationNode, [pleuID], tmpPleu, inputVolumeNode)
                    mergedArr = (
                        slicer.util.arrayFromVolume(tmpPeri).astype(bool) |
                        slicer.util.arrayFromVolume(tmpPleu).astype(bool)
                    ).astype(np.uint8)
                else:
                    mergedArr = slicer.util.arrayFromVolume(tmpPeri).astype(np.uint8)

                _reimportPleural(mergedArr, "Pericardial merge")
                seg.RemoveSegment(periID)

                for n in (tmpPeri, tmpPleu):
                    if n and n.GetScene():
                        slicer.mrmlScene.RemoveNode(n)

        # ─────────────────────────────────────────────────────────────────────────────
        # 🔹 3) SON TEMİZLİKLER
        # ─────────────────────────────────────────────────────────────────────────────
        _subtractTargetFromPleural()
        _subtractLobesFromPleural()
        
        # ---------------------------------------------------------------------------
        # 4)  SAHNE TEMİZLİĞİ – geçici label-map node ve segmentleri süpür
        # ---------------------------------------------------------------------------
        def _cleanupTempNodesAndSegments():
            """Temp/Merged/LabelMapVolume_* düğüm & segmentlerini sahneden siler."""
            # --- a) Geçici label-map düğümleri ------------------------------------
            for lmNode in slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode"):
                name = lmNode.GetName()
                if name.startswith(("Temp", "Merged_", "LabelMapVolume_")) \
                   and lmNode is not inputVolumeNode:
                    slicer.mrmlScene.RemoveNode(lmNode)

            # --- b) CombinedSegmentation içindeki gereksiz segmentler -------------
            badIDs = [sid for sid in seg.GetSegmentIDs()
                      if seg.GetSegment(sid).GetName().startswith(
                             ("Temp", "Merged_", "LabelMapVolume_"))]

            for sid in badIDs:
                seg.RemoveSegment(sid)

            self.logCallback("[DEBUG] Geçici düğüm ve segment temizliği yapıldı.")

        _cleanupTempNodesAndSegments()
        
        self.subtractNodulesFromLungVesselsClean(combinedSegmentationNode)


       
        # 5. Son olarak, CombinedSegmentation node’unu subject hierarchy’de input volume altında konumlandırın.      
        
        shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
        inputSHItem = shNode.GetItemByDataNode(inputVolumeNode)
        combinedSHItem = shNode.GetItemByDataNode(combinedSegmentationNode)
        if inputSHItem and combinedSHItem:
            shNode.SetItemParent(combinedSHItem, inputSHItem)
            self.logCallback("[DEBUG] CombinedSegmentation node, input volume altına taşındı.")
        else:
            self.logCallback("[WARNING] Subject hierarchy item'ları bulunamadı; CombinedSegmentation node parent'ı ayarlanamadı.")
            
        self.logCallback("[SUCCESS] Tüm görevler tamamlandı ve birleşik segmentation node oluşturuldu.")
        
        # 6. Artık tüm segmentasyon görevleri tamamlandı; şimdi birleşik node içindeki segmentlerin renklerini güncelleyelim. 
        segmentColorMapping = {
            # Arteriyel yapılar ve kalp ile ilgili:
            "aorta": (220, 20, 60),                    # Crimson
            "atrial_appendage_left": (205, 92, 92),      # Indian Red
            "heart": (255, 0, 0),                        # Kırmızı
            "common_carotid_artery_left": (220, 20, 60),   # Orange Red
            "common_carotid_artery_right": (220, 20, 60),
            "brachiocephalic_trunk": (220, 20, 60),
            "subclavian_artery_left": (220, 20, 60),
            "subclavian_artery_right": (220, 20, 60),
            "pulmonary_artery": (128, 0, 128),          # Mor
            "heart_myocardium": (255, 85, 130),
            "heart_atrium_left": (128, 0, 0),
            "heart_ventricle_left": (139, 0, 0),
            "heart_atrium_right": (128, 0, 32),
            "heart_ventricle_right": (101, 0, 11),
        
            # Venöz yapılar:
            "brachiocephalic_vein_left": (65, 105, 225),   # Royal Blue
            "brachiocephalic_vein_right": (65, 105, 225),
            "inferior_vena_cava": (30, 144, 255),          # Dodger Blue
            "superior_vena_cava": (30, 144, 255),
            "pulmonary_vein": (100, 149, 237),             # Cornflower Blue
    
            # Hava içeren yapılar (airway):
            "trachea": (0, 255, 0),                        # Yeşil
            "lung_trachea_bronchia": (34, 139, 34),         # Forest Green
    
            # Akciğer lobeleri (maskeler):
            "lung_lower_lobe_left": (176, 196, 222),        # Light Steel Blue
            "lung_lower_lobe_right": (176, 196, 222),
            "lung_middle_lobe_right": (172, 192, 218),
            "lung_upper_lobe_left": (174, 194, 220),
            "lung_upper_lobe_right": (174, 194, 220),
            "lung": (178, 198, 224),
    
            # Yuuşak Doku ile ilgili segmentler – hepsi aynı renkte olsun :
            "body_trunc": (255, 228, 196),            # Bisque
            "body_extremities": (255, 228, 196),            # Bisque
            "subcutaneous_fat": (255, 230, 198),
            "torso_fat": (254, 227, 195),
            "skeletal_muscle": (253, 226, 194),
            "autochthon_left": (253, 226, 194), 
            "autochthon_right": (253, 226, 194),
            "intermuscular_fat": (255, 229, 197),
    
            # Spleen: kırmızı tonlarından (örneğin tamamen kırmızı)
            "spleen": (255, 0, 0),
    
            # Özefagus ve Stomach: birbirine yakın tonlar (örneğin Turuncunun açık tonları)
            "esophagus": (255, 165, 0),  # Orange
            "stomach": (255, 140, 0),    # Aynı Orange tonu
            
            # Kemik yapılar
            "rib_left_1": (255, 255, 255), 
            "rib_left_2": (255, 255, 255), 
            "rib_left_3": (255, 255, 255), 
            "rib_left_4": (255, 255, 255), 
            "rib_left_5": (255, 255, 255),
            "rib_left_6": (255, 255, 255), 
            "rib_left_7": (255, 255, 255), 
            "rib_left_8": (255, 255, 255), 
            "rib_left_9": (255, 255, 255), 
            "rib_left_10": (255, 255, 255), 
            "rib_left_11": (255, 255, 255), 
            "rib_left_12": (255, 255, 255), 
            "rib_right_1": (255, 255, 255), 
            "rib_right_2": (255, 255, 255), 
            "rib_right_3": (255, 255, 255), 
            "rib_right_4": (255, 255, 255),
            "rib_right_5": (255, 255, 255), 
            "rib_right_6": (255, 255, 255), 
            "rib_right_7": (255, 255, 255), 
            "rib_right_8": (255, 255, 255), 
            "rib_right_9": (255, 255, 255), 
            "rib_right_10": (255, 255, 255), 
            "rib_right_11": (255, 255, 255), 
            "rib_right_12": (255, 255, 255), 
            "sternum": (255, 255, 255), 
            "costal_cartilages": (255, 255, 255),
            "vertebrae_T12": (255, 255, 255), 
            "vertebrae_T11": (255, 255, 255), 
            "vertebrae_T10": (255, 255, 255), 
            "vertebrae_T9": (255, 255, 255), 
            "vertebrae_T8": (255, 255, 255), 
            "vertebrae_T7": (255, 255, 255), 
            "vertebrae_T6": (255, 255, 255), 
            "vertebrae_T5": (255, 255, 255), 
            "vertebrae_T4": (255, 255, 255),  
            "vertebrae_T3": (255, 255, 255), 
            "vertebrae_T2": (255, 255, 255), 
            "vertebrae_T1": (255, 255, 255), 
            "vertebrae_C7": (255, 255, 255), 
            "scapula_left": (255, 255, 255), 
            "scapula_right": (255, 255, 255), 
            "clavicula_left": (255, 255, 255), 
            "clavicula_right": (255, 255, 255), 
            "humerus_left": (255, 255, 255), 
            "humerus_right": (255, 255, 255),
    
            # Diğer segmentler – örnek veya rastgele renkler:
            "lung_vessels": (128, 0, 128),   # mor
            "pericardial_effusion": (255, 153, 153),  # light red
            "pleural_effusion": (255, 182, 193),  # Light Pink
            "thyroid_gland": (255, 182, 193) # Light Pink 
        }
        
        combinedSegmentation = combinedSegmentationNode.GetSegmentation()
        for segID in combinedSegmentation.GetSegmentIDs():
            seg = combinedSegmentation.GetSegment(segID)
            segName = seg.GetName()
            if segName in segmentColorMapping:
                rgb = segmentColorMapping[segName]
                seg.SetColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                self.logCallback(f"[DEBUG] Segment '{segName}' rengi güncellendi: {rgb}")
            else:
                self.logCallback(f"[DEBUG] Segment '{segName}' için renk ataması yapılmadı.")

        # ───────────── Segment_* çöplerini temizle ─────────────
        segmentation = combinedSegmentationNode.GetSegmentation()

        segment_ids = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segment_ids)

        for idx in range(segment_ids.GetNumberOfValues()):
            sid = segment_ids.GetValue(idx)
            seg_name = segmentation.GetSegment(sid).GetName()
            if seg_name.startswith("Segment"):
                segmentation.RemoveSegment(sid)
                self.logCallback(f"[INFO] '{seg_name}' segmenti silindi.")

            
        #------------opasifikasyon--------------------------------------
        segNode  = slicer.util.getNode('CombinedSegmentation')
        dispNode = segNode.GetDisplayNode()

        opaque80   = {'TargetRegion'}
        transparent = {
            'body_extremities', 'lung', 'skeletal_muscle',
            'torso_fat', 'subcutaneous_fat'
        }

        seg = segNode.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            sid  = seg.GetNthSegmentID(i)
            name = seg.GetNthSegment(i).GetName()

            if name in opaque80:
                alpha = 0.80
            elif name in transparent:
                alpha = 0.0
            else:
                alpha = 0.40

            # 3-B kapalı-yüzey
            dispNode.SetSegmentOpacity3D(sid, alpha)
            # 2-B dolgu ve kontur
            dispNode.SetSegmentOpacity2DFill(sid,     alpha)
            dispNode.SetSegmentOpacity2DOutline(sid,  alpha)
            slicer.util.resetThreeDViews() 
        return combinedSegmentationNode

    
    
    
    
    def subtractNodulesFromLungVesselsClean(self, segmentationNode, vesselSegmentName="lung_vessels", noduleSegmentName="lung_nodules"):
        """
        lung_vessels segmentinden, lung_nodules ile çakışan alanları çıkarır.
        Segment doğrudan güncellenir. Referans geometri korunur. Temp node oluşturulmaz.
        """
        import numpy as np
        import slicer

        seg = segmentationNode.GetSegmentation()
        vesselSegmentID = seg.GetSegmentIdBySegmentName(vesselSegmentName)
        noduleSegmentID = seg.GetSegmentIdBySegmentName(noduleSegmentName)

        if not vesselSegmentID or not noduleSegmentID:
            print(f"[ERROR] Segmentler bulunamadı: {vesselSegmentName}, {noduleSegmentName}")
            return

        # Mevcut referans volume'u al (segmentasyonun bağlı olduğu)
        referenceVolumeID = seg.GetConversionParameter("ReferenceImageGeometry")
        if not referenceVolumeID:
            print("[ERROR] Referans hacim bulunamadı. BinaryLabelmap için referans eksik.")
            return
        referenceVolumeNode = slicer.mrmlScene.GetNodeByID(referenceVolumeID)

        # Labelmap temsilini güncelle ve arrayleri al
        segmentationNode.CreateBinaryLabelmapRepresentation()
        vesselsArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, vesselSegmentID, referenceVolumeNode)
        nodulesArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, noduleSegmentID, referenceVolumeNode)

        # Kesişimi çıkar (uint8 dönüşümü dikkatli yapılır)
        correctedArray = np.logical_and(vesselsArray, np.logical_not(nodulesArray)).astype(np.uint8)

        # Segmenti yerinde güncelle
        slicer.util.updateSegmentBinaryLabelmapFromArray(correctedArray, segmentationNode, vesselSegmentID, referenceVolumeNode)

        print(f"[INFO] '{vesselSegmentName}' segmenti güncellendi. Nodül çakışmaları çıkarıldı.")

    
    
    
    
    def setInputVolumeNode(self, node):
        """
        Set the input volume node and validate it.
        """
        if not node or not isinstance(node, slicer.vtkMRMLScalarVolumeNode):
            raise ValueError("Invalid input volume node. Please provide a valid vtkMRMLScalarVolumeNode.")
        if not node.GetImageData():
            raise ValueError("Input volume node has no image data.")
        self.inputVolumeNode = node
        print(f"[Python] Input volume node set: {node.GetName()} ({node.GetClassName()})")
 
    def getParameterNodeByName(self, nodeName: str):
        """
        Search for a parameter node by its name.
        """
        # Sahnedeki tüm ScriptedModule düğümlerini kontrol et
        for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLScriptedModuleNode"):
            if node.GetName() == nodeName:
                return node
        return None
    
    def getParameterNode(self):
        """
        Return the parameter node associated with this module.
        """
        node = self.getParameterNodeByName("LungBiopsyTractPlanner")
        if not node:
            # Eğer düğüm yoksa oluştur
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "LungBiopsyTractPlanner")
        return node              

    def __init__(self, widget=None, logCallback=None):
        """Widget referansı alır ve UI ile bağlantıyı sağlar."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.widget = widget  # UI widget bağlantısını sakla
        self.inputVolumeNode = None  # Initialize inputVolumeNode as None
        
        self.segmentationResults = {}   # Burada sonuçları saklayacağız
        self.outputQueue = queue.Queue(maxsize=0)
        
        # logCallback atanması
        if logCallback:
            self.logCallback = logCallback   # Mesajları widget tarafına iletecek
        else:
            self.logCallback = print        # widget verilmeyince doğrudan print
        
        
        # Görev-segment eşleştirmesi
        self.task_segment_map = {
            298: ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left", 
                "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", 
                "esophagus", "trachea", "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", 
                "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
                "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", 
                "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", 
                "vertebrae_C1", "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", 
                "common_carotid_artery_right", "common_carotid_artery_left", "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
                "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", 
                "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right",
                "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right",
                "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull", "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
                "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
                "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum", "costal_cartilages"],  # Total segmentler
            258: ["lung_vessels", "lung_trachea_bronchia"],  # Akciğer damarları ve bronşlar
            315: ["lung", "pleural_effusion", "pericardial_effusion"],  # Effüzyonlar
            481: ["subcutaneous_fat", "torso_fat", "skeletal_muscle"], 
            299: ["body_trunc", "body_extremities"], 
            301: ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "heart_atrium_right", "heart_ventricle_right", "aorta", "pulmonary_artery"],
            913: ["lung", "lung_nodules"],
        }

        # Görev adlarını task_id ile eşleştir
        self.task_name_map = {
            298: "total",  # Genel segmentasyon
            258: "lung_vessels",  # Akciğer damarları ve bronşlar
            315: "pleural_pericard_effusion",  # Plevral ve perikardiyal effüzyon
            481: "tissue_types", 
            299: "body",
            301: "heartchambers_highres",
            913: "lung_nodules"
        }
        

    def createEmphysemaSegment(self, inputVolumeNode, combinedSegmentationNode):
        """
        Emphysema/bulla segmentasyonu (TotalSegmentator segment node'dan, numpy mantığıyla çalışır):
        - TotalSegmentator lob segmentlerini numpy array olarak alır
        - Maksimum (OR) ile birleştirir (numpy seviyesinde)
        - CT ile çarpar, HU threshold yapar
        - Segmentation node üretir (reslice, import garantili)
        """

        self.widget.updateStatus("[INFO] Amfizem/Bulla segmentasyonu başlatılıyor...")

        # 1. Giriş hacmi kontrolü
        if not inputVolumeNode or not inputVolumeNode.GetImageData():
            slicer.util.errorDisplay("Geçerli bir giriş hacmi seçilmedi.")
            return None

        inputArray = slicer.util.arrayFromVolume(inputVolumeNode)
        segmentation = combinedSegmentationNode.GetSegmentation()

        # 2. Lob segment ID'lerini bul (lung geçenler)
        lobeSegmentIDs = []
        for i in range(segmentation.GetNumberOfSegments()):
            segName = segmentation.GetNthSegment(i).GetName()
            if "lobe" in segName.lower():
                segID = segmentation.GetSegmentIdBySegmentName(segName)
                lobeSegmentIDs.append(segID)

        print(f"[INFO] Bulunan lob segment ID'leri: {lobeSegmentIDs}")

        if not lobeSegmentIDs:
            slicer.util.errorDisplay("Lob segmentleri bulunamadı.")
            return None

        # 3. Tüm lob maskelerini numpy array olarak topla ve OR (maximum) ile birleştir
        combinedLungMask = None
        for segID in lobeSegmentIDs:
            tmpLabelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TmpLobeMask")
            idArray = vtk.vtkStringArray()
            idArray.InsertNextValue(segID)

            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(
               combinedSegmentationNode, idArray, tmpLabelmapNode, inputVolumeNode
            )

            lobMaskArray = slicer.util.arrayFromVolume(tmpLabelmapNode)
            slicer.mrmlScene.RemoveNode(tmpLabelmapNode)

            if combinedLungMask is None:
                combinedLungMask = lobMaskArray.copy()
            else:
                combinedLungMask = np.maximum(combinedLungMask, lobMaskArray)

        print(f"[INFO] Combined Lung Mask voxels: {np.count_nonzero(combinedLungMask)}")
        self.lungMaskNumpy = combinedLungMask.copy()   # numpy binary (0/1)
        self._lungKD = None                            # KD-tree henüz yok      

        # 4. CT ile çarp → sadece akciğer HU
        lungHUArray = inputArray * combinedLungMask

        # 5. HU threshold (amfizem aralığı) → -1000 ile -930 HU arasında
        emphysemaMask = np.zeros_like(lungHUArray, dtype=np.uint8)
        emphysemaMask[(combinedLungMask > 0) & (lungHUArray >= -1000) & (lungHUArray <= -930)] = 1

        print(f"[INFO] Amfizem Mask voxels after threshold: {np.count_nonzero(emphysemaMask)}")

        if np.count_nonzero(emphysemaMask) == 0:
            slicer.util.errorDisplay("Amfizem aralığında voxel bulunamadı. Segment oluşmadı.")
            return None

        # 6. Maskeyi volume node olarak oluştur → segmentation'a import edilecek
        emphysemaLMNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "EmphysemaLM")
        slicer.util.updateVolumeFromArray(emphysemaLMNode, emphysemaMask)
        emphysemaLMNode.SetOrigin(inputVolumeNode.GetOrigin())
        emphysemaLMNode.SetSpacing(inputVolumeNode.GetSpacing())
        m = vtk.vtkMatrix4x4()
        inputVolumeNode.GetIJKToRASMatrix(m)
        emphysemaLMNode.SetIJKToRASMatrix(m)

        # 7. Segmentation node oluştur ve import et
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "EmphysemaSegmentation")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(emphysemaLMNode, segmentationNode)
        slicer.mrmlScene.RemoveNode(emphysemaLMNode)

        # 8. Segment adı ve renk ayarla
        segment = segmentationNode.GetSegmentation().GetNthSegment(0)
        if segment:
            segment.SetName("emphysema_bulla")
            segment.SetColor(0.0, 0.0, 139 / 255.0)

        print("Amfizem segmentasyonu tamamlandı. Segment sayısı:",
              segmentationNode.GetSegmentation().GetNumberOfSegments())

        return segmentationNode

    def initializeLungMask(self, segmentationNode, volumeNode):
        print("initializeLungMask başlıyor")
        self.segmentationNode = segmentationNode
        self.inputVolumeNode = volumeNode
        segmentationNode.GetSegmentation().SetConversionParameter("ReferenceImageGeometry", volumeNode.GetID())
        segmentationNode.CreateBinaryLabelmapRepresentation()
        
        segmentation = segmentationNode.GetSegmentation()
        lobe_masks = [
            mask for (_, mask) in iter_valid_lobes(segmentationNode,
                                                   volumeNode,
                                                   return_mask=True)
            if mask is not None and np.any(mask)
        ]
        if lobe_masks:                                    # ≥1 geçerli maske
            lungMask = np.logical_or.reduce(lobe_masks).astype(np.uint8)
        else:                                             # hiç maske yoksa → 0
            self.logCallback("[WARN] Hiç geçerli akciğer lobu bulunamadı!")
            lungMask = np.zeros(slicer.util.arrayFromVolume(volumeNode).shape,
                                dtype=np.uint8)

        self.lungMaskNumpy = lungMask

        # KD-tree başlangıçta boş
        self.ijkToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(self.ijkToRAS)
        self._lungKD = None
            
    def lungParenchymaDistance(self, p_start, p_end):
        """
        İki RAS noktası arasındaki çizgi boyunca kaç mm akciğer parankimi içinde ilerleniyor?
        self.lungMaskNumpy ve self.inputVolumeNode.GetSpacing()'ı kullanır.
        """
        # KD-tree cache
        if self._lungKD is None:
            # voxel indeksi (z,y,x) -> RAS (x,y,z)
            vox = np.argwhere(self.lungMaskNumpy > 0)
            ijkHomogeneous = np.column_stack((vox[:,2], vox[:,1], vox[:,0], np.ones(len(vox))))
            ijkToRAS_np = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    ijkToRAS_np[i, j] = self.ijkToRAS.GetElement(i, j)
            rasCoords = (ijkToRAS_np @ ijkHomogeneous.T).T[:, :3]
            
            self._lungKD = cKDTree(rasCoords)

        # şimdi çizgide örnekleme
        p0 = np.array(p_start)
        p1 = np.array(p_end)
        L = np.linalg.norm(p1 - p0)
        if L == 0:
            return 0.0
        step_mm = 2.0
        steps = max(int(L/step_mm), 1)
        vec = (p1 - p0) / steps
        mids = p0 + vec*(np.arange(steps)[:,None] + 0.5)  # adım ortaları
        inside = self._lungKD.query_ball_point(mids, r=1.0)
        inside_mm = sum(bool(lst) for lst in inside) * np.linalg.norm(vec)
        return inside_mm        
            
    def _ensureLungKD(self):
        """
        self._lungKD None ise bir kez oluşturur.
        initializeLungMask içinde hazır veriler mevcut varsayılır.
        """
        if not hasattr(self, "lungMaskNumpy"):
            self.initializeLungMask(self.segmentationNode, self.inputVolumeNode)
        
        if self._lungKD is not None:        # zaten hazır
            return

        # Maskede hiç voxel kalmamışsa boş KD-tree kur
        vox = np.argwhere(self.lungMaskNumpy > 0)
        if vox.size == 0:
            from scipy.spatial import cKDTree
            self._lungKD = cKDTree(np.empty((0, 3)))
            return

        # IJK → RAS dönüşümü
        ijkHom = np.column_stack((vox[:, 2], vox[:, 1], vox[:, 0],
                                  np.ones(len(vox))))
        M = np.array([[self.ijkToRAS.GetElement(i, j)
                       for j in range(4)] for i in range(4)])
        ras = (M @ ijkHom.T).T[:, :3]

        from scipy.spatial import cKDTree
        self._lungKD = cKDTree(ras)
    
 
    # ─────────────────────────────────────────────────────────────
    #  Posterior projection overlap checker
    # ─────────────────────────────────────────────────────────────
    def projects_on_scapulae_posterior(
        self,
        segmentation_node,
        volume_node,
        *,
        target_name: str = "TargetRegion",
        scap_right_name: str = "scapula_right",
        scap_left_name: str = "scapula_left",
        restrict_to_scap_z: bool = True,
        ) -> bool:
        """
        Returns True if the posterior (Y-axis) projection of *target_name*
        intersects either scapula mask; otherwise False.

        • If neither scapula segment exists *or* both masks are empty, returns False.
        • If the target segment doesn’t exist, raises ValueError (critical).
        """
        import numpy as np, slicer

        seg = segmentation_node.GetSegmentation()
        bin_rep = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
        seg.SetConversionParameter("ReferenceImageGeometry", volume_node.GetID())
        if not seg.ContainsRepresentation(bin_rep):
            segmentation_node.CreateBinaryLabelmapRepresentation()

        # ── Resolve segment IDs ──
        tid = seg.GetSegmentIdBySegmentName(target_name)
        if not tid:
            raise ValueError(f"Target segment “{target_name}” not found.")

        sid_r = seg.GetSegmentIdBySegmentName(scap_right_name)
        sid_l = seg.GetSegmentIdBySegmentName(scap_left_name)

        # ── Pull scapula masks that actually exist ──
        scap_masks = []
        if sid_r:
            mask_r = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, sid_r, volume_node
            ).astype(bool)
            if np.any(mask_r):                       # non-empty check
                scap_masks.append(mask_r)

        if sid_l:
            mask_l = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentation_node, sid_l, volume_node
            ).astype(bool)
            if np.any(mask_l):
                scap_masks.append(mask_l)

        if not scap_masks:
            # neither scapula present or both empty → no possible overlap
            return False

        scap_mask = np.logical_or.reduce(scap_masks)

        # ── Target mask ──
        tar_mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            segmentation_node, tid, volume_node
        ).astype(bool)

        # ── Optional Z-window clipping ──
        if restrict_to_scap_z:
            z_idx = np.where(scap_mask)[0]
            zmin, zmax = int(z_idx.min()), int(z_idx.max())
            tar_mask = tar_mask[zmin : zmax + 1]
            scap_mask = scap_mask[zmin : zmax + 1]

        # ── Posterior projection and intersection test ──
        tar_proj = np.any(tar_mask, axis=1)     # collapse Y → (Z, X)
        scap_proj = np.any(scap_mask, axis=1)

        return bool(np.any(tar_proj & scap_proj))


    def analyzeAndVisualizeTracts(self, segmentationNode, combinedSegmentationNode, riskSegments):
        """
        Calculates candidate biopsy tracts, scores their risks/difficul-
        ties, and draws the selected lines in the scene.

        """
        print("Analiz düğmesine basıldı")
        vtk.vtkObject.GlobalWarningDisplayOn() 
 
        def _safeIntersect(obb, p0, p1):
            try:
                if obb is None:
                    return 0
                if np.allclose(p0, p1):     # aynı nokta ⇒ IntersectWithLine çöker
                    print("[WARN] p0==p1 SKIP")
                    return 0
                pts, ids = vtk.vtkPoints(), vtk.vtkIdList()
                r = obb.IntersectWithLine(list(map(float, p0)),
                                          list(map(float, p1)),
                                          pts, ids)
                return r
            except Exception as e:
                print("[EXC] IntersectWithLine:", e)   # Python hatası olursa
                return 0
                
 

        # ------------------------------------------------------------
        #  Parametreler ve sabitler
        # ------------------------------------------------------------
        SEG_NODE_NAME           = "CombinedSegmentation"
        TARGET_SEG_NAME         = "TargetRegion"
        BODY_SEG_NAME           = "body_trunc"

        SUBSAMPLE_EVERY_NTH     = 500
        MIN_DIST_BETWEEN_POINTS = 10.0   # mm
        NEEDLE_LEN_MAX          = 190    # mm
        MIN_INSIDE_MM           = 10     # mm
        allowedSegments = [
            "lung_upper_lobe_left","lung_lower_lobe_left",
            "lung_upper_lobe_right","lung_lower_lobe_right",
            "lung_middle_lobe_right","pleural_effusion", 
            "subcutaneous_fat","torso_fat","skeletal_muscle",
            "autochthon_left","autochthon_right","body_trunc",
            "emphysema_bulla","TargetRegion","lung", "lung_nodules"
        ]


        
        segNode = slicer.util.getNode(SEG_NODE_NAME)
        segNode.CreateClosedSurfaceRepresentation()
        volNode = self.widget.ui.inputSelector.currentNode()
        seg     = segNode.GetSegmentation()
        segNode.GetSegmentation().SetConversionParameter("ReferenceImageGeometry", volNode.GetID())
        segNode.CreateBinaryLabelmapRepresentation()
        problem_segments = []
        emptySegIDs      = set()     
        for i in range(seg.GetNumberOfSegments()):
            segName = seg.GetNthSegment(i).GetName()
            segId = seg.GetNthSegmentID(i)
            poly = vtk.vtkPolyData()
            segNode.GetClosedSurfaceRepresentation(segId, poly)
            if poly is None or poly.GetNumberOfPoints() == 0:
                problem_segments.append(segName)
                emptySegIDs.add(segId) 
                print(f"[UYARI] {segName} segmentinde kapalı yüzey yok veya yüzey noktası sıfır!")

        if problem_segments:
            self.logCallback(
                "[INFO] Boş/kapalı-yüzeysiz segmentler analizden çıkarıldı: "
                + ", ".join(problem_segments)
            )

            
        print("initializeLungMask başlatılacak")       
        self.initializeLungMask(segNode, volNode)
        try:
            # Konsol çıktılarının GUI’ye de düşmesi için isteğe bağlı log
            if self.widget:
                self.widget.updateStatus("[INFO] Trakt analizi başlatıldı…")
   
            # ───────────────────────────────────────────────
            # 1) Derinlik-bağımlı sürekli OR fonksiyonu
            # ───────────────────────────────────────────────
            
            
            
            
            def depth_or_continuous(mm: float) -> float:
                anchors = [(00, 0.0),                       # log(OR)=0
                           (20, math.log(2.16)),            # 0.77
                           (30, math.log(2.38)),            # 0.87
                           (50, math.log(8.47))]            # 2.14
                
                for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
                    if x0 <= mm <= x1:
                        t = (mm - x0) / (x1 - x0)
                        return math.exp(y0 + t * (y1 - y0))

                # >50 mm  → son segmentin eğimiyle extrapolasyon
                slope = (anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0] - anchors[-2][0])
                log_or = anchors[-1][1] + slope * (mm - anchors[-1][0])
                return math.exp(log_or)
                
            def depth_or_continuous_hmr(mm: float) -> float:
                anchors = [(0, 0.0),                      
                           (30, math.log(4.558)),            
                           (50, math.log(25.641))]            

                for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
                    if x0 <= mm <= x1:
                        t = (mm - x0) / (x1 - x0)
                        return math.exp(y0 + t * (y1 - y0))

                # >50 mm  → son segmentin eğimiyle extrapolasyon
                slope = (anchors[-1][1] - anchors[-2][1]) / (anchors[-1][0] - anchors[-2][0])
                log_or = anchors[-1][1] + slope * (mm - anchors[-1][0])
                return math.exp(log_or)    
            



            # 1) TargetRegion maskesi
            print("TargetRegion maskesi başlatılacak")  
            targetId   = seg.GetSegmentIdBySegmentName(TARGET_SEG_NAME)
            polyTar = vtk.vtkPolyData()
            segNode.GetClosedSurfaceRepresentation(targetId, polyTar)
            tar_points = polyTar.GetPoints()
            if tar_points is None or tar_points.GetNumberOfPoints() == 0:
                slicer.util.errorDisplay("TargetRegion segmentinin yüzeyinde hiç nokta yok! Segmenti veya kapalı yüzeyini kontrol edin.")
                return
            targetMask = self.getSegmentMaskAsArray(segNode, targetId, volNode)
            print("TargetRegion maskesi tamamlandı")  

            # 2) Lezyon sol alt lobda mı? (bunu hesaplar ve diğerlerini etkiler)
            print("lezyon konum tespiti")
            llId    = seg.GetSegmentIdBySegmentName("lung_lower_lobe_left")
            llMask = self.getSegmentMaskAsArray(segNode, llId, volNode)
            left_lower_lobe = 1 if np.any(targetMask & llMask) else 0

            # 3) Lezyon sol alt lobdaysa → sağ alt lob feature kesin 0
            if left_lower_lobe:
                right_lower_lobe = 0
            else:
                rlId   = seg.GetSegmentIdBySegmentName("lung_lower_lobe_right")
                rlMask = self.getSegmentMaskAsArray(segNode, rlId, volNode)
                right_lower_lobe = 1 if np.any(targetMask & rlMask) else 0

            # 4) Lezyon sol alt lobdaysa → right_hilar da 0, değilse sırayla hesapla
            if left_lower_lobe:
                right_hilar = 0
            else:
                # 4a) Lezyon sağ akciğerde mi?
                # Sağ akciğer loblarının isim kümesi
                RIGHT_LOBES = {
                    "lung_upper_lobe_right",
                    "lung_middle_lobe_right",
                    "lung_lower_lobe_right",
                }

                # iter_valid_lobes(…, as_ids=True)  →  Tuple(name, segId) değerleri üretir
                ids_right = [
                    sid
                    for name, sid in iter_valid_lobes(segNode, volNode, as_ids=True)
                    if name in RIGHT_LOBES
                ]
                combined_right = np.zeros_like(targetMask, dtype=bool)
                
                for sid in ids_right:
                    combined_right |= self.getSegmentMaskAsArray(segNode, sid, volNode).astype(bool)
                lesion_in_right_lung = int(np.any(targetMask & combined_right))

                # 4b) Eğer değilse direkt 0; eğer evetse mesafe hesapla
                if not lesion_in_right_lung:
                    right_hilar = 0
                else:
                    # ensure closed‐surface repr.
                    if not seg.ContainsRepresentation("Closed surface"):
                        segNode.CreateClosedSurfaceRepresentation()

                    polyPA  = vtk.vtkPolyData()
                    segNode.GetClosedSurfaceRepresentation(
                        seg.GetSegmentIdBySegmentName("pulmonary_artery"), polyPA)
                    pts, n = tar_points, tar_points.GetNumberOfPoints()
                    step, minD2 = max(1, n//1000), float("inf")
                    locatorPA = vtk.vtkPointLocator()
                    locatorPA.SetDataSet(polyPA)
                    locatorPA.BuildLocator()
                    for i in range(0, n, step):
                        p = pts.GetPoint(i)
                        closest_id = locatorPA.FindClosestPoint(p)  # Sadece 1 argüman!
                        closest_point = polyPA.GetPoint(closest_id)
                        d2 = np.sum((np.array(p) - np.array(closest_point))**2)
                        if d2 < minD2:
                            minD2 = d2
                    right_hilar = 1 if math.sqrt(minD2) <= 30.0 else 0
            print("lezyon konum tespiti tamamlandı")
             
            # 5) TargetRegion en geniş çapı ≤30 mm mı?
            print("lezyon en geniş çapı tespiti")
          
            impTar = vtk.vtkImplicitPolyDataDistance();  impTar.SetInput(polyTar)
            bounds = polyTar.GetBounds()
            diameters = [bounds[1]-bounds[0],
                         bounds[3]-bounds[2],
                         bounds[5]-bounds[4]]
            max_diam = max(diameters)
            size_le3cm = 1 if max_diam <= 30.0 else 0
            size_2_3cm = 1 if (20.0 < max_diam <= 30.0) else 0
            size_3_4cm = 1 if (30.0 < max_diam <= 40.0) else 0
            size_4_5cm = 1 if (40.0 < max_diam <= 50.0) else 0
            size_5cm = 1 if (50.0 < max_diam ) else 0
            print("lezyon en geniş çapı tespiti tamamlandı")

            # 6) TargetRegion ile fat/muscle arası minDist hesapla (sadece bir kez)
            print("lezyon akciger derinligi tespiti")
            appendFat = vtk.vtkAppendPolyData()

            segmentation = segNode.GetSegmentation()
            segmentIDs = vtk.vtkStringArray()
            segmentation.GetSegmentIDs(segmentIDs)

            for i in range(segmentIDs.GetNumberOfValues()):
                sid = segmentIDs.GetValue(i)
                name = segmentation.GetSegment(sid).GetName()
                # Sabit isimler veya adı "rib" veya "costa" ile başlayan segmentleri al
                if name in ("subcutaneous_fat", "skeletal_muscle", "torso_fat") or \
                   name.lower().startswith("rib") or name.lower().startswith("costa"):
                    poly = vtk.vtkPolyData()
                    segNode.GetClosedSurfaceRepresentation(sid, poly)
                    appendFat.AddInputData(poly)

            appendFat.Update()
            polyFat = appendFat.GetOutput()

            # Eğer polyFat’da hiç nokta yoksa hedef ile plevra mesafesini "sonsuz" kabul et
            if polyFat.GetNumberOfPoints() == 0:
                minDist = float("inf")          # → pleural_contact = 0
            else:
                locatorFat = vtk.vtkPointLocator()
                locatorFat.SetDataSet(polyFat)
                locatorFat.BuildLocator()
                ptsTar = polyTar.GetPoints()
                nTar   = ptsTar.GetNumberOfPoints()
                step   = max(1, nTar // 1000)   # her ~1000 noktada bir örnekle
                minD2  = float("inf")

                for j in range(0, nTar, step):
                    p = np.array(ptsTar.GetPoint(j))
                    cid = locatorFat.FindClosestPoint(p)        # *** SADECE 1 argüman ***
                    if cid >= 0:
                        cp  = np.array(polyFat.GetPoint(cid))
                        d2  = np.sum((p - cp) ** 2)
                        if d2 < minD2:
                            minD2 = d2

                minDist = math.sqrt(minD2)

            pleural_contact = 1 if minDist < 1.0 else 0
            print("lezyon akciger derinligi tespiti tamamlandı, plevra teması tespiti başlıyor.")

            
            # --- Plevra ile temas alanı (mm²)  ---------------------------------
            
            if pleural_contact == 1:
                polyBody = vtk.vtkPolyData()
                segNode.GetClosedSurfaceRepresentation(
                        seg.GetSegmentIdBySegmentName(BODY_SEG_NAME), polyBody)
                        
                
                intf = vtk.vtkIntersectionPolyDataFilter()
                intf.SetInputData(0, polyTar)
                intf.SetInputData(1, polyBody)
                intf.Update()
                contactPoly = intf.GetOutput()

                contactArea = 0.0
                cells  = contactPoly.GetPolys()
                pts    = contactPoly.GetPoints()
                idList = vtk.vtkIdList()
                for _ in range(cells.GetNumberOfCells()):
                    cells.GetNextCell(idList)
                    if idList.GetNumberOfIds() != 3:
                        continue
                    p0 = np.array(pts.GetPoint(idList.GetId(0)))
                    p1 = np.array(pts.GetPoint(idList.GetId(1)))
                    p2 = np.array(pts.GetPoint(idList.GetId(2)))
                    contactArea += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

                pleural_patch_too_small = contactArea < 2.0   # ≤ 2 mm² ise “küçük” de  

                # Küçükse line‐OBB kısıtını uygula
                if pleural_patch_too_small:
                    obbTiny = vtk.vtkOBBTree()
                    obbTiny.SetDataSet(contactPoly)
                    obbTiny.BuildLocator()
                else:
                    obbTiny = None
            else:
                obbTiny = None
                
            print("plevra teması tespiti tamamlandı, PHT tespiti başlıyor. ")


            # 8) MPAD/AAD >1 mi?  – aynı çizgide ölçüm
            paId = seg.GetSegmentIdBySegmentName("pulmonary_artery")
            aoId = seg.GetSegmentIdBySegmentName("aorta")
            pa_mask = self.getSegmentMaskAsArray(segNode, paId, volNode)
            ao_mask = self.getSegmentMaskAsArray(segNode, aoId, volNode)
            # a) En geniş PA alanlı slice
            counts = pa_mask.reshape(pa_mask.shape[0], -1).sum(axis=1)
            k_max = int(np.argmax(counts))
            pa_slice = pa_mask[k_max]; ao_slice = ao_mask[k_max]
            # b) Aorta’nın en geniş yatay satırı
            rows = ao_slice.shape[0]
            best_w = -1; best_y = None; xmin = xmax = None
            for y in range(rows):
                xs = np.where(ao_slice[y])[0]
                if xs.size<2: continue
                w = xs.max() - xs.min()
                if w>best_w:
                    best_w = w; best_y = y; xmin, xmax = xs.min(), xs.max()
            ao_width = best_w * volNode.GetSpacing()[0]
            # c) Aynı satırda PA çapını ölç
            xs_pa = np.where(pa_slice[best_y])[0]
            pa_width = (xs_pa.max()-xs_pa.min()) * volNode.GetSpacing()[0] if xs_pa.size>=2 else 0.0
            # d) Oran ve binary sonucu
            mpad_aad_gt1 = 1 if (pa_width/ao_width)>1.0 else 0
            
            print("HT tespiti tamamlandı, amfizemli lob tspiti başlıyor. ")
            # 9) TargetRegion ile kesişen loblarda emphysema_bulla var mı?
            emId = seg.GetSegmentIdBySegmentName("emphysema_bulla")
            emMask = self.getSegmentMaskAsArray(segNode, emId, volNode) if emId else None

            emphysema = 0
            if emMask is not None and np.any(emMask):
                # SADECE geçerli loblar:
                for _, lobeMask in iter_valid_lobes(segNode, volNode, return_mask=True):
                    if np.any(targetMask & lobeMask) and np.any(emMask & lobeMask):
                        emphysema = 1
                        break

            print("Amfizemli lob tespiti tamamladnı, ipsilateral effüzyon tespiti başlıyor ")
            
            # 10) Ipsilateral pleural effusion var mı?
            peId = seg.GetSegmentIdBySegmentName("pleural_effusion")
            if peId is None:
                ipsilateral_effusion = 0
            else:
                segNode.CreateBinaryLabelmapRepresentation()      # (idempotent ⇒ defalarca çağrılabilir)
                
                from vtk import vtkStringArray
                ids = vtkStringArray()
                ids.InsertNextValue(peId)

                # --- 2. Temsil volNode geometrisine uygun mu? ---------------------
                logic = slicer.modules.segmentations.logic()
                tmpLabel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

                ok = logic.ExportSegmentsToLabelmapNode(
                        segNode,                      # kaynak segmentation
                        ids,
                        tmpLabel,                     # hedef label-map
                        volNode)                      # referans geometri

                if (not ok or       # hâlâ üretilemiyorsa → boş maske kabul et
                    tmpLabel.GetImageData() is None or                    # ↙ hiçbir voxel yoksa
                    tmpLabel.GetImageData().GetPointData().GetScalars() is None):
                    ipsilateral_effusion = 0
                    slicer.mrmlScene.RemoveNode(tmpLabel)  
                else:
                    peMask = slicer.util.arrayFromVolume(tmpLabel)
                    slicer.mrmlScene.RemoveNode(tmpLabel)          # sahneyi temizle

                    # --- 3. Ipsilateral mi? -------------------------------------------------
                    nx       = peMask.shape[2]
                    mid_x    = nx / 2.0
                    coords   = np.argwhere(targetMask > 0)
                    mean_x   = coords[:, 2].mean() if coords.size else mid_x

                    if mean_x < mid_x:       # lezyon sol tarafta
                        ipsilateral_effusion = 1 if np.any(peMask[:, :, :int(mid_x)]) else 0
                    else:                    # lezyon sağ tarafta
                        ipsilateral_effusion = 1 if np.any(peMask[:, :, int(mid_x):]) else 0


            print("İpsilateral effüzyon tespiti tamamlandı ")
            
            # Tüm lezyon-temelli özellikleri hazırla
            lesionFeatures = {
                "left_lower_lobe": left_lower_lobe, 
                "right_lower_lobe": right_lower_lobe, 
                "right_hilar": right_hilar, 
                "size_le3cm": size_le3cm, 
                "size_2_3cm": size_2_3cm, 
                "size_3_4cm": size_3_4cm, 
                "size_4_5cm": size_4_5cm, 
                "size_5cm": size_5cm,                
                "pleural_contact": pleural_contact,  
                "mpad_aad_gt1": mpad_aad_gt1, 
                "emphysema": emphysema, 
                "ipsilateral_effusion": ipsilateral_effusion }


            # --- 1) Risk tablosu ve compute_risk fonksiyonu ------------
            RISK_TABLE = {
                "hemorrhage": {
                    # — Any-grade PH modeli (Zhu et al. 2020) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
                    #https://doi.org/10.21037/qims-19-1024
                    "any_grade": {
                        "base": -0.795,   # intercept β₀
                        "factors": {
                            "left_lower_lobe": {
                                "OR": 1.948, "CI": (1.209, 3.138), "p": 0.006
                            #targetregion segmenti, left_lower_lobe ile kesişiyorsa 1, kesişmiyorsa 0.(trakt bağımsız).
                            },
                            "right_lower_lobe": {
                                "OR": 1.754, "CI": (1.125, 2.734), "p": 0.013
                            #targetregion segmenti, right_lower_lobe ile kesişiyorsa 1, kesişmiyorsa 0.(trakt bağımsız).
                            },
                            "right_hilar": {
                                "OR": 5.368, "CI": (1.518, 18.986), "p": 0.009
                            #targetregion segmenti yüzeyi ile pulmonary_Artery segmenti yüzeylerinin birbirlerine en yakın noktaları arasındaki mesafe 3 cm'den azsa 1, değilse 0. (trakt bağımsız).
                            },
                            "size_le3cm": {
                                "OR": 1.628, "CI": (1.186, 2.236), "p": 0.003
                            #targetregion segmentinin en geniş çapı 3cm'den azsa 1, değilse 0. (trakt bağımsız).
                            },
                            
                            "depth_cont_hmr": {"beta": 1.0},
                            #aşağıdaki paremetrelere göre fonksiyonla hespalanıyor
                                #"depth_3_5cm": {
                                #    "OR": 4.558, "CI": (2.141, 9.704), "p": "<0.001"
                                #trakın akciğer parenkiminde katettiği mesafe 3 - 5 cm arasında mı?(trakt bağımlı).
                                #},
                                #"depth_5cm": {
                                #    "OR": 25.641, "CI": (12.276, 53.560), "p": "<0.001"
                                #trakın akciğer parenkiminde katettiği mesafe 5 cm'den fazla mı?(trakt bağımlı).
                                #},
                            "lung_metastases": {
                                "OR": 6.695, "CI": (2.618, 17.122), "p": "<0.001"
                            #lung_metastases segmentinde birden fazla kapalı yüzey var mı? (trakt bağımsız).
                            }
                        }
                    },
                    # — Higher-grade PH modeli (Zhu et al. 2020) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
                    #https://doi.org/10.21037/qims-19-1024
                    "high_grade": {
                        "base": -2.590,  # intercept β₀ 
                        "factors": {
                            "mpad_aad_gt1": {
                                "OR": 1.871, "CI": (1.063, 3.294), "p": 0.03
                            #aksiyel kesitlerde pulmoner arter segmentinin en çok alana sahip olduğu slice'ı tespit edip. Bu slice'ta pulmoner arter segmentinin sağında kalan aort segmentinin sağ-sol istikametinde çapı ile bu çapı ile aynı hizade pulmoner arter segmentinin çapını ölçüp oranlaya bilirsin. (trakt bağımsız).
                            },
                            "size_le3cm": {
                                "OR": 1.769, "CI": (1.081, 2.897), "p": 0.023 
                            #targetregion segmentinin en geniş çapı 3cm'den azsa 1, değilse 0.(trakt bağımsız).
                            },
                            
                            "depth_cont_hmr": {"beta": 1.0},
                            #aşağıdaki paremetrelere göre fonksiyonla hespalanıyor
                            #"depth_5cm": {
                            #    "OR": 5.880, "CI": (2.046, 16.898), "p": 0.001
                            #targetregion segmentinin dış yüzeyi ile subcutaneous_Fat veya skeletal_muscle segmentine en yakın mesafesi 5 cm'den fazla mı? (trakt bağımsız).
                            #},
                            
                            "emphysema": {
                                "OR": 2.810, "CI": (1.709, 4.621), "p": "<0.001"
                            #targetregion ile kesişen akciğer loblarında amfizem segmenti var mı? (trakt bağımsız).
                            },
                            "lung_metastases": {
                                "OR": 6.687, "CI": (2.629, 17.011), "p": "<0.001"
                            #lung_metastases segmentinde birden fazla kapalı yüzey var mı? (trakt bağımsız).
                            }
                        }
                    }
                },

                "pneumothorax": {
                    # — Genel PX modeli (Huo et al. 2020 Table 4) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
                    #https://doi.org/10.1259/bjr.20190866
                    "general": {
                        "base": -0.887,  # intercept β₀
                        "factors": {

                            "crossing_pts": {
                                "OR": 2.566, "CI": (1.71, 3.851), "p": "<0.001"
                            #"source": "Deng et al., BMC Pulm Med 2024",
                            #"doi": "10.1186/s12890-024-03307-z"
                            },  
                            
                            "anterior_entry": {
                                "OR": 1.83, "CI": (1.51, 2.21), "p": "<0.001"
                            },  
                            #vücuda giriş noktaları anteiorda olan traktlar için risk faktörü (trakt bağımlı)

                            "lateral_entry": {
                                "OR": 1.89, "CI": (0.43, 8.33), "p": ""
                            },
                            #vücuda giriş noktaları anteiorda olan traktlar için risk faktörü (trakt bağımlı)

                            "bulla_crossed": {
                                "OR": 6.13, "CI": (3.73, 10.06), "p": "<0.001"
                            },  # :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
                            #trakt amfizem segmentinden geçiyor mu? (trakt bağımlı).

                            "fissure_crossed": {
                                "OR": 3.75, "CI": (2.57, 5.46), "p": "<0.001"
                            },  # :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
                            #trakt 1'den fazla lobdan geçiyor mu? (trakt bağımlı).

                            "size_2_3cm": {
                                "OR": 0.5, "CI": (0.4, 0.64), "p": "<0.001"
                            },  # meta-analiz :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
                            #targetregion segmentinin en geniş çapı 4cm'den azsa 1, değilse 0. (trakt bağımsız).

                            "size_3_4cm": {
                                "OR": 0.58, "CI": (0.5, 0.67), "p": "<0.001"
                            },  # meta-analiz :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
                            #targetregion segmentinin en geniş çapı 4cm'den azsa 1, değilse 0. (trakt bağımsız).

                            "size_4_5cm": {
                                "OR": 0.48, "CI": (0.30, 0.75), "p": "<0.01"
                            },  # meta-analiz :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
                            #targetregion segmentinin en geniş çapı 4cm'den azsa 1, değilse 0. (trakt bağımsız).

                            "size_5cm": {
                                "OR": 0.34, "CI": (0.18, 0.65), "p": "<0.01"
                            },  # meta-analiz :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
                            #targetregion segmentinin en geniş çapı 4cm'den azsa 1, değilse 0. (trakt bağımsız).

                            
                            "depth_cont": {"beta": 1.0},
                            #aşağıdaki paremetrelere göre fonksiyonla hespalanıyor
                                #"depth_2_3cm": {
                                #    "OR": 2.16, "CI": (1.31, 3.57), "p": "<0.01"
                                #},  # meta-analiz :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
                                #traktın akciğer parenkimindeki uzunluğu 3 cm'den fazla mı? (trakt bağımlı).

                                #"depth_3_5cm": {
                                #    "OR": 2.38, "CI": (1.60, 3.53), "p": "<0.001"
                                #},  # meta-analiz :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
                                #traktın akciğer parenkimindeki uzunluğu 3 ile 5 cm arasında mı? (trakt bağımlı).

                                #"depth_5cm": {
                                #    "OR": 8.47, "CI": (3.44, 20.9), "p": "<0.001"
                                #},  # meta-analiz :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
                                #traktın akciğer parenkimindeki uzunluğu 5 cm'den fazla mı? (trakt bağımlı).

                            "pleural_contact": {
                                "OR": 0.57, "CI": (0.39, 0.85), "p": "<0.01"
                            },
                            #targetregion segmentinin dış yüzeyi ile subcutaneous_Fat veya skeletal_muscle segmentine en yakın mesafesi 1 mm'den az mı? (trakt bağımsız).
                           
                            "pleural_fluid_instillation": {
                                "OR": 0.071, "p": 0.003
                            },  
                            # Brönnimann et al. 2024 :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
                            #https://doi.org/10.1016/j.ejrad.2024.111529
                            #trakt pleural_Effusion segmentinden geçiyor mu? (trakt bağımlı).
                        
                            "ipsilateral_effusion": {
                                "OR": 0.65,  "p": 0.05,
                            }  
                            # Anil et al. 2022 :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
                            #https://doi.org/10.1016/j.jacr.2022.04.010
                            #targetregion ile aynı tarafta pleural_effusion segmenti var mı? (traktan bağımsız).

                        }
                    },
                    # — Drain gerektiren PX (Huo et al. 2020 Table 5) :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
                    #https://doi.org/10.1259/bjr.20190866
                    "drain_required": {
                        "base": -2.300,    # intercept β₀ 
                        "factors": {
                            
                            "anterior_entry": {
                                "OR": 1.94, "CI": (1.62, 2.32), "p": "<0.001"
                            },  
                            #vücuda giriş noktaları anteiorda olan traktlar için risk faktörü (trakt bağımlı)

                            "lateral_entry": {
                                "OR": 1.19, "CI": (0.9, 1.56), "p": "<0.05"
                            },
                            #vücuda giriş noktaları lateralde olan traktlar için risk faktörü (trakt bağımlı)

                            "fissure_crossed": {
                                "OR": 3.54, "CI": (2.32, 5.40), "p": "<0.05"
                            },
                            #trakt 1'den fazla lobdan geçiyor mu? (burada trakta bağımlı).
                            "bulla_crossed": {
                                "OR": 11.04, "CI": (5.32, 22.90), "p": "<0.05"
                            },
                            #trakt amfizem segmentinden geçiyor mu? (burada trakta bağımlı).
                            "emphysema": {
                                "OR": 6.44, "CI": (4.27, 9.72), "p": "<0.01"
                            },

                            "pleural_contact": {
                                "OR": 0.53, "CI": (0.32, 0.87), "p": "<0.01"
                            },
                            #targetregion segmentinin dış yüzeyi ile subcutaneous_Fat veya skeletal_muscle segmentine en yakın mesafesi 1 mm'den az mı? (trakt bağımsız).
                            
                            "pleural_fluid_instillation": {
                                "OR": 0.071, "p": 0.003
                            },  
                            # Brönnimann et al. 2024 :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
                            #https://doi.org/10.1016/j.ejrad.2024.111529
                            #trakt pleural_Effusion segmentinden geçiyor mu? (trakt bağımlı).
                          
                            "ipsilateral_effusion": {
                                "OR": 0.48, "p": 0.05
                            }  
                            # Anil et al. 2022 :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}
                            #https://doi.org/10.1016/j.jacr.2022.04.010
                            #trakt pleural_Effusion segmentinden geçiyor mu? (trakt bağımlı).
                        }    
                    }
                    
                }
            }
            
            PX_TRACT_FACTORS = {
                "anterior_entry", "lateral_entry", "fissure_crossed",
                "bulla_crossed", "depth_cont",
                "pleural_fluid_instillation", "    maskArray = maskArrayUInt8.astype(np.uint8, copy=False)"
            }
            HMR_TRACT_FACTORS = {
                "depth_cont_hmr"
            }
            

            
            for evt in RISK_TABLE.values():
                for mdl in evt.values():
                    for f in mdl["factors"].values():
                        if "OR" in f:
                            f["beta"] = math.log(f["OR"])


            def compute_risk_tract(event, model, tract_feats, base_eta):
                """
                base_eta : Lezyon-bağımlı sabit logit (önceden hesaplandı)
                tract_feats : {factor_name: value, ...}
                """
                table = RISK_TABLE[event][model]
                eta   = base_eta

                for name, value in tract_feats.items():
                    factor = table["factors"].get(name)
                    if factor:                     # sadece tabloda varsa ekle
                        eta += factor["beta"] * value

                return 1.0 / (1.0 + math.exp(-eta))

            # --- 2) Stub feature fonksiyonları -------------------------

            
           
            

            def feature_large_vessel_dist_mm(start_point, end_point, segNode, volumeNode):
                imp = _getVesselImplicit(segNode)
                if imp is None:
                    return 0.0        # damar bulunamadı → risk 0
                    
                p0, p1 = np.array(start_point), np.array(end_point)
                vec    = p1 - p0
                L      = np.linalg.norm(vec)
                if L == 0: return 0
                # 10-mm aralıklarla örnekle
                step_mm = 10.0
                steps   = max(int(L/step_mm), 1)        # en az 1 örnek
                mids    = [p0 + vec*( (i+0.5)/steps ) for i in range(steps)]
                dists = np.abs([imp.EvaluateFunction(pt) for pt in mids])
                score = 0
                for d in dists:
                    if   d < 5 :  score += 5
                    elif d < 10:  score += 4
                    elif d < 15:  score += 3
                    elif d < 20:  score += 2
                    elif d < 25:  score += 1
                return score              


            def feature_bulla_crossed(start_point, end_point, segNode, volumeNode):
                """
                Traktın emphysema_bulla segmentini kesip kesmediğini kontrol eder.
                Kesiyorsa 1, değilse 0 döner.
                """
                seg = segNode.GetSegmentation()
                bullaId = seg.GetSegmentIdBySegmentName("emphysema_bulla")
                if not bullaId:
                    return 0

                polyBulla = _getClosed(segNode, bullaId)

                
                if polyBulla is None:
                    return 0
                obb = vtk.vtkOBBTree()
                obb.SetDataSet(polyBulla); obb.BuildLocator()
                return 1 if _safeIntersect(obb, start_point, end_point) > 0 else 0

            # --- trakt-bağımlı: fissure crossed (1 if >1 lobe) ---
            def feature_fissure_crossed(p0, p1, segNode, _):
                if np.allclose(p0, p1):          # <-- EK
                    return 0
                crossed = 0
                for _, sid in iter_valid_lobes(segNode, volNode, as_ids=True):
                    poly = _getClosed(segNode, sid)
                    if poly is None or poly.GetNumberOfPoints()==0:
                        continue
                    obb = vtk.vtkOBBTree(); obb.SetDataSet(poly); obb.BuildLocator()
                    if _safeIntersect(obb, p0, p1):
                        crossed += 1
                        if crossed > 1:
                            return 1
                return 0
    
    
               

            def feature_pleural_fluid_instillation(start_point, end_point, segNode, volumeNode):
                """
                Traktın hiçbir akciğer lob segmentini kesmeksizin [LUNG_SEGMENTS] pleural_effusion segmentini kesip kesmediğini kontrol eder.
                HİÇBİR AKCİĞER LOB SEGMENTİNİ GEÇMEYİP kesiyorsa 1, değilse 0 döner.
                """
                if np.allclose(p0, p1):           # <-- EK
                    return 0
                seg = segNode.GetSegmentation()

                # 1) Önce akciğer loblarıyla kesişim var mı diye bak
                for _, sid in iter_valid_lobes(segNode, volNode, as_ids=True):
                    poly_lob = _getClosed(segNode, sid)
                    if poly_lob is None:
                        continue
                    obb_lob = vtk.vtkOBBTree(); obb_lob.SetDataSet(poly_lob); obb_lob.BuildLocator()
                    if _safeIntersect(obb_lob, start_point, end_point) > 0:
                        return 0    

                # 2) Akciğer lobuna rastlamadıysa pleural_effusion segmentini kontrol et
                pe_id = seg.GetSegmentIdBySegmentName("pleural_effusion")
                if not pe_id:
                    return 0
                poly_pe = _getClosed(segNode, pe_id)
                if poly_pe is None or poly_pe.GetNumberOfPoints()==0:
                    return 0

                obb_pe = vtk.vtkOBBTree()
                obb_pe.SetDataSet(poly_pe)
                obb_pe.BuildLocator()
                # Pleural effusion ile kesişim varsa 1, yoksa 0 döner
                return 1 if _safeIntersect(obb_pe, start_point, end_point) > 0 else 0

            
            def feature_technic_diff(start_point, end_point, segNode, volumeNode):
                """
                Traktın geçtiği aksiyel slice sayısını döndürür (Z ekseninde).
                """
                rasToIJK = vtk.vtkMatrix4x4()
                volumeNode.GetRASToIJKMatrix(rasToIJK)

                def ras_to_ijk(ras_point):
                    rasVec = list(ras_point) + [1.0]
                    ijkVec = [0.0, 0.0, 0.0, 0.0]
                    rasToIJK.MultiplyPoint(rasVec, ijkVec)
                    return [int(round(c)) for c in ijkVec[:3]]

                ijk_start = ras_to_ijk(start_point)
                ijk_end   = ras_to_ijk(end_point)
                z_start = ijk_start[2]
                z_end   = ijk_end[2]
                return abs(z_end - z_start) + 1

            def feature_anterior_entry(start_point, end_point, segNode, volumeNode):
                """
                1 döner eğer entry noktası body_trunc segmentinin anterior yüzeyine
                (Y+) en yakınsa. Diğer durumlar için 0.
                """


                # skin segmentini al, closed surface temsili oluştur
                seg = segNode.GetSegmentation()
                sid = seg.GetSegmentIdBySegmentName("body_trunc")
                if not sid:
                    print("[ERROR] 'body_trunc' segmenti yok.")
                    return 0
                if not seg.ContainsRepresentation("Closed surface"):
                    segNode.CreateClosedSurfaceRepresentation()
                poly = vtk.vtkPolyData()
                segNode.GetClosedSurfaceRepresentation(sid, poly)

                # bounding box elde et: (xmin,xmax, ymin,ymax, zmin,zmax)
                xmin, xmax, ymin, ymax, zmin, zmax = poly.GetBounds()
                p0 = np.array(start_point, dtype=float)

                # yüzeylere uzaklıklar
                dist_ant  = ymax - p0[1]
                dist_post = p0[1] - ymin
                dist_lat  = min(p0[0] - xmin, xmax - p0[0])

                # en küçük uzaklığa bak: anterior mı?
                return int(dist_ant < dist_lat and dist_ant < dist_post)

            def feature_lateral_entry(start_point, end_point, segNode, volumeNode):
                """
                1 döner eğer entry noktası body_trunc segmentinin
                lateral (X ekseni ±) yüzeyine en yakınsa. Diğer durumlar için 0.
                """
 
                

                seg = segNode.GetSegmentation()
                sid = seg.GetSegmentIdBySegmentName("body_trunc")
                if not sid:
                    print("[ERROR] 'body_trunc' segmenti yok.")
                    return 0
                if not seg.ContainsRepresentation("Closed surface"):
                    segNode.CreateClosedSurfaceRepresentation()
                poly = vtk.vtkPolyData()
                segNode.GetClosedSurfaceRepresentation(sid, poly)

                xmin, xmax, ymin, ymax, zmin, zmax = poly.GetBounds()
                p0 = np.array(start_point, dtype=float)

                dist_ant  = ymax - p0[1]
                dist_post = p0[1] - ymin
                dist_lat  = min(p0[0] - xmin, xmax - p0[0])

                # en küçük uzaklığa bak: lateral mı?
                return int(dist_lat < dist_ant and dist_lat < dist_post)



            # Map factor adı → fonksiyon
            feature_funcs = {
                "large_vessel_dist_mm": feature_large_vessel_dist_mm,
                "bulla_crossed": feature_bulla_crossed,
                "fissure_crossed": feature_fissure_crossed,
                "pleural_fluid_instillation": feature_pleural_fluid_instillation,
                "anterior_entry": feature_anterior_entry,
                "lateral_entry": feature_lateral_entry,
                "technic_diff": feature_technic_diff,
            }

            
            print("cilt yüzeyi tespiti başlıyor ")
            # --- 3) Sahneden nodları al, cilt yüzeyi oluştur ------------

            segNode = slicer.util.getNode(SEG_NODE_NAME)
            seg = segNode.GetSegmentation()
            if not seg.ContainsRepresentation("Closed surface"):
                segNode.CreateClosedSurfaceRepresentation()
            polySkin = vtk.vtkPolyData()
            segNode.GetClosedSurfaceRepresentation(
                seg.GetSegmentIdBySegmentName(BODY_SEG_NAME), polySkin)
                
                
            dec = vtk.vtkDecimatePro(); dec.SetInputData(polySkin); dec.SetTargetReduction(0.75)
            dec.PreserveTopologyOn(); dec.Update()
            polyReduced = dec.GetOutput()
            normGen = vtk.vtkPolyDataNormals(); normGen.SetInputData(polyReduced)
            normGen.ComputePointNormalsOn(); normGen.SplittingOff(); normGen.Update()
            polyWithNormals = normGen.GetOutput()
            normArr = polyWithNormals.GetPointData().GetNormals() 
            
            N = polyWithNormals.GetNumberOfPoints()

            # VTK→NumPy kopyaları (hızlı!)
            pts_np   = np.array([polyWithNormals.GetPoint(i)  for i in range(N)])
            norms_np = np.array([normArr.GetTuple(i)          for i in range(N)])

            # ---------- Üst / alt yüzey maskesi ------------------------
            max_angle_deg = 30.0
            cos_thr       = math.cos(math.radians(max_angle_deg))
            norms_unit    = norms_np / np.linalg.norm(norms_np, axis=1, keepdims=True)

            sup_cand  =  norms_unit[:, 2] > +cos_thr
            inf_cand  =  norms_unit[:, 2] < -cos_thr
            side_cand = ~(sup_cand | inf_cand)
            
            # 2) Superior tamamen düz mü? .........................................
            FLAT_TOL_MM = 2.0
            is_sup_flat = False
            if sup_cand.any():
                z_span = pts_np[sup_cand, 2].ptp()
                is_sup_flat = z_span < FLAT_TOL_MM  
                
            # 3) Boyun maskesi – omuz genişliğine göre adaptif ....................
            neck_mask = np.zeros_like(sup_cand, dtype=bool)
            if not is_sup_flat and sup_cand.any():          # yalnız “üst” düzensizse
                x_sup  = pts_np[sup_cand, 0]
                x_mid  = np.median(x_sup)
                # Omuz “tam” genişliği ≈ 2×95p
                w95    = 2.0 * np.percentile(np.abs(x_sup - x_mid), 95)
                neck_half_w = 0.35 * w95                    # %35’i: ~boyun genişliği
                neck_mask   = sup_cand & (np.abs(pts_np[:,0] - x_mid) < neck_half_w)                

            # 4) Nihai keep_mask ..................................................
            keep_mask            = np.copy(side_cand)       # yan yüzeyler serbest
            keep_mask[inf_cand]  = False                    # K-3
            if is_sup_flat:                                 # K-1
                keep_mask[sup_cand] = False
            else:                                           # K-2
                keep_mask[neck_mask] = False                # yalnız boyun platosu yasak
                




            # Son seçim + seyrekleştirme
            skin_pts = pts_np[keep_mask][::SUBSAMPLE_EVERY_NTH]
            normals  = norms_np[keep_mask][::SUBSAMPLE_EVERY_NTH]
 
            pointLocator = vtk.vtkPointLocator()
            pointLocator.SetDataSet(polyWithNormals)
            pointLocator.BuildLocator()
            
            print("cilt yüzeyi tespiti tamamlandı ")



            print("Forbidden OBB tespiti başlıyor.  ")
            # --- 4) Target & forbidden OBB hazırlıkları --------------

            locTar = vtk.vtkCellLocator(); locTar.SetDataSet(polyTar); locTar.BuildLocator()
            obbTar = vtk.vtkOBBTree();     obbTar.SetDataSet(polyTar); obbTar.BuildLocator()
                        
            appendForb = vtk.vtkAppendPolyData()
            for i in range(seg.GetNumberOfSegments()):
                name = seg.GetNthSegment(i).GetName()
                if name not in allowedSegments:
                    sid = seg.GetNthSegmentID(i)
                    if not sid:
                        continue
                    poly = vtk.vtkPolyData()
                    segNode.GetClosedSurfaceRepresentation(sid, poly)
                    if poly is not None and poly.GetNumberOfPoints() > 0:
                        appendForb.AddInputData(poly)
                    
            appendForb.Update()
            polyForb = appendForb.GetOutput()
            if polyForb.GetNumberOfPoints() > 0:
                obbForb = vtk.vtkOBBTree(); obbForb.SetDataSet(polyForb); obbForb.BuildLocator()
            else:
                obbForb = None

            locSkin = vtk.vtkCellLocator()
            locSkin.SetDataSet(polySkin)
            locSkin.BuildLocator()
            forbiddenOBB = obbForb
            
            print("Forbidden OBB tespiti tamamlandı.  ")
            
            
            
            # ------------------------------------------------------------------
            #  Lezyon-bağımlı faktörlerin toplam logit (eta) değeri
            # ------------------------------------------------------------------
            # 0️⃣ Checkbox kontrolü: Eğer işaret yoksa ‘lung_metastases’ anahtarını çıkar
            if not self.widget.metastasisCheckBox.isChecked():
                RISK_TABLE["hemorrhage"]["any_grade"]["factors"].pop("lung_metastases", None)
                RISK_TABLE["hemorrhage"]["high_grade"]["factors"].pop("lung_metastases", None)
            
            
            
            RISK_KEYS = {
                'ph_any'  : ("hemorrhage",  "any_grade"),
                'ph_high' : ("hemorrhage",  "high_grade"),
                'px_full' : ("pneumothorax","general"),
                'px_drain': ("pneumothorax","drain_required"),
            }

            LESION_BASE_ETA = {}   # {key: eta0}

            for key, (event, model) in RISK_KEYS.items():
                table = RISK_TABLE[event][model]
                eta   = table["base"]

                for name, value in lesionFeatures.items():
                    factor = table["factors"].get(name)
                    if factor:
                        beta = factor.get("beta", math.log(factor["OR"]))
                        eta += factor["beta"] * value     # lezyon katkıları

                LESION_BASE_ETA[key] = eta
                
            #---- scapula projection ----    
            #seg   = slicer.util.getNode("CombinedSegmentation")
            #scap_overlap = self.projects_on_scapulae_posterior(segNode, volNode)

            # --- 5) Aday traktları topla ------------------------------
            print("Aday trakt toplanması başlıyor  ")
 
            print(f"[DBG] skin_pts: {len(skin_pts)}   normals: {len(normals)}")

            candidates = []
            p_entry = [0.0,0.0,0.0]
            p0 = [0.0,0.0,0.0]; p1 = [0.0,0.0,0.0]
            pts = vtk.vtkPoints(); ids = vtk.vtkIdList() 
            print("Aday trakt toplanması batching başlıyor  ")
            ALL_RISK_FACTORS = list(PX_TRACT_FACTORS | HMR_TRACT_FACTORS)
            batch_size = 300
            for batch_start in range(0, len(skin_pts), batch_size):
                batch_pts = skin_pts[batch_start:batch_start+batch_size]
                batch_nrms = normals[batch_start:batch_start+batch_size]
                
                dbg_i = 0
                for ras, nrm in zip(batch_pts, batch_nrms):
                    dbg_i += 1
                    print(f"[DBG] cand{dbg_i}  ras={ras.tolist()}"); sys.stdout.flush()
                    
                    # --- p0'ı orijinal cilt yüzeyine projeksiyon ---
                    skin_entry = [0.0,0.0,0.0]
                    cid, subId, d2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
                    locSkin.FindClosestPoint(ras, skin_entry, cid, subId, d2)
                    p0 = np.array(skin_entry, dtype=float)    # (ras yerine p0 ismi net)
                    ras = p0 
                    
                    self._ensureLungKD()                                # KD-tree garanti hazır
                    if self._lungKD.query_ball_point(p0, r=1.0):   # 1 mm içinde voxel yoksa
                        continue
                    
                    # --- En yakın TargetRegion noktasını bul ---
                    cid, subId, dist2 = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
                    locTar.FindClosestPoint(p0, p_entry, cid, subId, dist2)

                    d = math.sqrt(dist2) 
                    vec = np.array(p_entry) - p0
                    L = np.linalg.norm(vec)
                    if L==0 or ((L+MIN_INSIDE_MM)>NEEDLE_LEN_MAX): 
                        continue
                        
                    dirV  = vec/L
                    nrm_norm = nrm/np.linalg.norm(nrm)
                    theta = math.degrees(math.acos(abs(max(min(np.dot(nrm_norm, dirV),1),-1))))
                    if theta>45:                                 # ❷ ort açı sınırı
                        continue

                    p1 = (np.array(p_entry)+dirV*MIN_INSIDE_MM).tolist()

                    # ❸ yasak yüzey-kontrolleri
                    
                    if _safeIntersect(obbTar, p0, p1)<1:    
                        continue
                    if obbForb is not None and _safeIntersect(obbForb, p0, p1) > 0:     
                        continue
                    if impTar.EvaluateFunction(p1)>0: 
                        continue
                    cross_flag = 0
                    if obbTiny is not None and _safeIntersect(obbTiny, p0, p1) > 0: 
                        cross_flag = 1
                        continue

                    # ❹ akciğer-içi mesafe
                    #lung_mm = self.lungParenchymaDistance(p0, p_entry)
                    distance_to_lung, closest_lung_idx = self._lungKD.query(p0)
                    lung_mm = max(d - distance_to_lung, 0.0)
                    lung_mm = lung_mm if lung_mm > 0 else 0.0
                    

                    # --- ❺ Özellik seti ---------------------------------------------------
                    features = { name: func(p0, p_entry, segNode, volNode)
                                 for name, func in feature_funcs.items() }

                    features["depth_cont"] = math.log(depth_or_continuous(lung_mm))
                    features["crossing_pts"] = cross_flag 
                    
                    log_or = math.log(depth_or_continuous_hmr(lung_mm))
                    if lung_mm > 50:  # 5 cm'den büyükse
                        features["depth_cont_hmr"] = log_or / 1.473
                    else:
                        features["depth_cont_hmr"] = log_or


                    # ❻ Riskler
                    px  = compute_risk_tract("pneumothorax","general",
                                            {k:v for k,v in features.items() if k in PX_TRACT_FACTORS}, LESION_BASE_ETA["px_full"])
                    hmr  = compute_risk_tract("hemorrhage","any_grade",
                                            {k:v for k,v in features.items() if k in HMR_TRACT_FACTORS}, LESION_BASE_ETA["ph_any"]) 
                    px_t  = compute_risk_tract("pneumothorax","drain_required",
                                            {k:v for k,v in features.items() if k in PX_TRACT_FACTORS}, LESION_BASE_ETA["px_drain"])
                                            
                    hmr_h = compute_risk_tract("hemorrhage","high_grade",
                                            {k:v for k,v in features.items() if k in HMR_TRACT_FACTORS}, LESION_BASE_ETA["ph_high"])                        
                                           
                    p_end = p_entry+dirV*MIN_INSIDE_MM
                    diff_score = feature_technic_diff(ras, p_end, segNode, volNode)
 
                    candidates.append((p0, p_end, lung_mm, px, hmr, px_t, hmr_h, d, theta, diff_score, features.copy()))
    
            print(f"[INFO] Aday trakt sayısı: {len(candidates)}")
            import pandas as pd

            df = pd.DataFrame([
                {
                    "ras": ras,
                    "p_end": p_end,
                    "lung_mm": lung_mm,
                    "px_score": px,
                    "hmr_score": hmr,
                    "px_t_score": px_t,
                    "hmr_h_score": hmr_h,
                    "target_dist": d,
                    "theta": theta,
                    "diff_score": diff_score,
                    **{f"risk_{k}": features.get(k, 0) for k in ALL_RISK_FACTORS},
                    "features": features
                }
                for ras, p_end, lung_mm, px, hmr, px_t, hmr_h, d, theta, diff_score, features in candidates
            ])

            # ----------------------------------------------------------------------
            # 1) Px skoruna göre ana quantile etiketi (0 = Q1 en düşük risk)
            # ----------------------------------------------------------------------
            uniq = df["px_score"].nunique()
            bins = min(4, uniq)                 # en çok 4, ama uniq kadar
            labels = list(range(bins))          # [0], [0,1], [0,1,2] veya [0,1,2,3]
            df["px_q"] = pd.qcut(
                df["px_score"], q=bins, labels=labels,
                duplicates="drop"
            ).astype(int)                       # tek bin varsa hepsi 0

            # ----------------------------------------------------------------------
            # 2) Klinik grup (0=A … 4=E)
            # ----------------------------------------------------------------------
            cond_A = (df["lung_mm"] < 3) & (df["risk_pleural_fluid_instillation"] == 1) & (df["risk_lateral_entry"] == 0)
            cond_B = (df["lung_mm"] < 3) & (df["risk_pleural_fluid_instillation"] == 1) & (df["risk_lateral_entry"] == 1)
            cond_C = (df["lung_mm"] < 3) & (df["risk_pleural_fluid_instillation"] == 0) & (df["risk_lateral_entry"] == 0)
            cond_D = (df["lung_mm"] < 3) & (df["risk_pleural_fluid_instillation"] == 0) & (df["risk_lateral_entry"] == 1)

            df["clin_grp"] = np.select(
                [cond_A, cond_B, cond_C, cond_D],
                [0,      1,      2,      3],
                default=4                                        # E grubu
            )

            # ----------------------------------------------------------------------
            # 3)  Sadece E-gruplarında depth_cont_hmr’yi 4 alt-quantile’a böl
            #     (her px_q içinde).  Uyarıları kapatmak için observed=False,
            #     apply sırasında include_groups=False kullanılıyor.
            # ----------------------------------------------------------------------
            df["depth_q"] = 0
            def _add_depth_q(sub):
                if sub["risk_depth_cont_hmr"].nunique() >= 4:
                    sub["depth_q"] = pd.qcut(
                        sub["risk_depth_cont_hmr"], 4,
                        labels=[0, 1, 2, 3]
                    )
                return sub

            mask_E = df["clin_grp"] == 4
            df.loc[mask_E] = (
                df[mask_E]
                  .groupby("px_q", observed=False, group_keys=False)
                  .apply(_add_depth_q, include_groups=False)
            )

            # ----------------------------------------------------------------------
            # 4)  Lateral giriş önceliği (0 = lateral_entry yok → önde)
            # ----------------------------------------------------------------------
            df["lat_rank"] = np.where(df["risk_lateral_entry"] == 0, 0, 1)

            # 5)  A-D gruplarında px_score;  E gruplarında depth_q
            df["prim_sort"] = df["px_score"]
            df.loc[mask_E, "prim_sort"] = df.loc[mask_E, "depth_q"].astype(int)

            # ----------------------------------------------------------------------
            # 6)  Nihai öncelik sırası
            # ----------------------------------------------------------------------
            df_sorted = df.sort_values(
                ["px_q",        # Q1 < Q2 < Q3 < Q4
                 "clin_grp",    # A < B < C < D < E
                 "prim_sort",   # A-D: px   |  E: depth_q
                 "lat_rank",    # L0 önde
                 "diff_score",
                 "target_dist"]
            ).reset_index(drop=True)

            # ----------------------------------------------------------------------
            # 7)  İlk 10 trakt
            # ----------------------------------------------------------------------
            top10_df = df_sorted.head(10)

            #selected = [
            #    (row["ras"], row["p_end"], row["lung_mm"], row["px_score"], row["hmr_score"],
            #     row["px_t_score"], row["hmr_h_score"], row["target_dist"], row["theta"],
            #     row["diff_score"], row["features"])
            #    for _, row in top10_df.iterrows()
            #]
            #selected_tracts = {"pneumothorax": selected}
            
            top10_df = top10_df.assign(
                position = np.select(
                    [
                        top10_df["risk_lateral_entry"] == 1,
                        top10_df["risk_anterior_entry"] == 1,
                        scap_overlap
                    ],
                    [
                        "Lateral Decubitus",          # 1. koşul
                        "Supine",                     # 2. koşul
                        "Prone, consider Scapular Mobilizations"  # 3. koşul
                    ],
                    default="Prone"
                )
)


            # --- 10) Zorluk skorunu hesapla, sırala ve çiz ----
            tracts = []
            for _, row in top10_df.iterrows():
                ras = row["ras"]
                p_end = row["p_end"]
                lung_mm = row["lung_mm"]
                px = row["px_score"]
                hmr = row["hmr_score"]
                px_t = row["px_t_score"]
                hmr_h = row["hmr_h_score"]
                d = row["target_dist"]
                theta = row["theta"]
                diff_score = row["diff_score"]
                features = row["features"]

                LVD_score = feature_large_vessel_dist_mm(ras, p_end, segNode, volNode)
                p_total  = 1 - (1-hmr)*(1-px)
                modality_recommendation = "consider US" if (lung_mm < 5) and (d < 100) else "CT"
                L = d

                tracts.append({
                    'ras': ras,
                    'p_end': p_end,
                    'overall': p_total,
                    'ph_any': hmr,
                    'px': px,
                    'ph_high': hmr_h,
                    'px_drain': px_t,
                    'diff': diff_score,
                    'LVD': LVD_score,
                    'mod': modality_recommendation,
                    'L': L,
                    "position": row["position"] 
                })


            def assign_ranks(tracts, key, rank_key):
                for rank, t in enumerate(sorted(tracts, key=lambda x: x[key], reverse=False), start=1):
                    t[rank_key] = rank
            assign_ranks(tracts, 'px',   'rank_px')
            assign_ranks(tracts, 'L',   'L')
            assign_ranks(tracts, 'diff',     'rank_diff')
            assign_ranks(tracts, 'overall', 'rank_overall')
            assign_ranks(tracts, 'ph_any',   'rank_ph_any')

            tracts_sorted_by_risk = sorted(tracts, key=lambda x: (x['px'], x['L'], x['diff'], x['overall'], x['ph_any']))
            num_tracts = len(tracts_sorted_by_risk)


            for idx, t in enumerate(tracts_sorted_by_risk, start=1):

                name = f"Tract_{idx}"
                pct = lambda p: f"{p*100:.1f}%"
                tract_length_mm = np.linalg.norm(t['p_end'] - t['ras'])
                tract_length_cm = int(round(tract_length_mm / 10.0))
                Overall_risk     = pct(t['overall'])
                Overall_rank     = f"rank {t['rank_overall']}"
                Hmr              = pct(t['ph_any'])
                Hmr_rank         = f"rank {t['rank_ph_any']}"
                Hmr_high         = pct(t['ph_high'])
                Px               = pct(t['px'])
                Px_rank          = f"rank {t['rank_px']}"
                Px_high          = pct(t['px_drain'])
                Difficulty       = f"{int(t['diff'])}"
                Diff_rank        = f"rank {t['rank_diff']}"
                LVD              = f"{int(t['LVD'])}"
                Modality         = t['mod']
                position     = t.get("position", "-")   
                
           
                desc = (
                    f"Ovr         : {Overall_risk} ({Overall_rank})\n"
                    f"Px          : {Px} ({Px_high}) ({Px_rank})\n"
                    f"Hmr         : {Hmr} ({Hmr_high}) ({Hmr_rank})\n"
                    f"Dif         : {Difficulty} ({Diff_rank})\n"
                    f"Len         : {tract_length_cm} cm \n"
                    f"LVD         : {LVD} \n"    
                    f"Mod         : {Modality}"
                    f"Pos         : {position}" 
                )

                line = slicer.mrmlScene.GetFirstNodeByName(name) or \
                       slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsLineNode', name)
                line.RemoveAllControlPoints()
                line.AddControlPoint(t['ras'].tolist())
                line.AddControlPoint(t['p_end'].tolist())
                line.SetDescription(desc)
                line.SetNthControlPointDescription(0, desc)   # ilk kontrol noktası
                line.SetNthControlPointDescription(1, "")     # ikincisini boş bırak
                label_full = (
                    f"Tr_{idx}\n"
                    f"Ovr:{Overall_risk} ({Overall_rank})\n"
                    f"Px :{Px} ({Px_high}) ({Px_rank})\n"
                    f"Hmr:{Hmr} ({Hmr_high}) ({Hmr_rank})\n"
                    f"Dif:{Difficulty} ({Diff_rank})\n"
                    f"Len:{tract_length_cm} cm \n"
                    f"LVD:{LVD} \n"
                    f"Mod:{Modality}\n"
                    f"Pos:{position}"
                )

                line.SetName(label_full)
                line.SetSelected(True)
                
                risk_norm = (idx - 1) / (num_tracts - 1) if num_tracts > 1 else 0
                r = risk_norm
                g = 1.0 - risk_norm
                b = 0.0


                disp = line.GetDisplayNode()
                disp.SetSelectedColor(risk_norm, 1.0 - risk_norm, 0.0)
                disp.SetLineThickness(0.3)
                disp.SetTextScale(2)
                disp.SetPointLabelsVisibility(True)
                disp.SetPropertiesLabelVisibility(False)
                line.SetNthControlPointLabel(0, label_full)
                line.SetNthControlPointLabel(1, "")
                print(f"[DEBUG] {name} description set:\n{desc}")
                print(f"[DEBUG] {name}: overall_risk%={Overall_risk}, overall_rank={Overall_rank}, px_risk%={Px}, px_rank={Px_rank}, Hmr_risk%={Hmr}, hmr_rank={Hmr_rank}, diff={Difficulty}, df_rank={Diff_rank}, L={tract_length_cm} cm, LVD={LVD}, Mod={Modality}") 


            self.processManualTractL(segmentationNode, combinedSegmentationNode, ALL_RISK_FACTORS)
        except Exception as e:
            slicer.util.errorDisplay(f"Trakt analizi sırasında hata: {e}")
             
            import traceback; traceback.print_exc()
            return None
            
#
# -----------------------------------------------------------------------------
#  TESTS – LungBiopsyTractPlannerTest
# -----------------------------------------------------------------------------
#


    
class LungBiopsyTractPlannerTest(ScriptedLoadableModuleTest):
    """
    Hızlı doğrulama: modül yüklenebiliyor ve logic nesnesi alınabiliyor mu?
    Ek veri indirme / segmentasyon yapılmaz; sadece ≈1 s sürer.
    """

    def setUp(self):
        """Her testten önce sahneyi temizle."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Tek adımlık smoke test."""
        self.delayDisplay("🚦  Smoke test başlıyor …")
        logic = slicer.util.getModuleLogic("LungBiopsyTractPlanner")
        self.assertIsNotNone(
            logic,
            "❌  LungBiopsyTractPlanner logic nesnesi yüklenemedi!"
        )
        self.delayDisplay("✅  Modül logic nesnesi başarıyla alındı.")

