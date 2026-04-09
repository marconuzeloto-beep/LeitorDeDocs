"""
API do sistema LeitorDeDocs.
Utiliza Ollama com modelos Llama locais para extração de documentos.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_provider import OllamaIndisponivel, OllamaProvider, conectar
from document_processor import TIPOS_DOCUMENTOS, processar_documento
from models import RespostaProcessamento, ResultadoArquivo

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
EXTENSOES_PERMITIDAS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

provider: Optional[OllamaProvider] = None
ia_disponivel: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider, ia_disponivel

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    modelo   = os.getenv("OLLAMA_MODEL", "llama3.2-vision").strip()

    try:
        provider = conectar(base_url, modelo)
        ia_disponivel = True
        logger.info(f"✅ {provider.nome} pronto")
    except OllamaIndisponivel as e:
        logger.error(f"❌ Ollama indisponível:\n{e}")
        ia_disponivel = False

    yield
    logger.info("Servidor encerrado")


app = FastAPI(
    title="LeitorDeDocs API",
    description="Extração inteligente de documentos com Ollama / Llama local",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def raiz():
    return {
        "status": "online",
        "provedor": provider.nome if provider else None,
        "ia_disponivel": ia_disponivel,
        "aviso": None if ia_disponivel else (
            "Ollama não está configurado. "
            "Instale o Ollama (https://ollama.com), execute 'ollama pull llama3.2-vision' "
            "e reinicie o servidor."
        )
    }


@app.get("/status")
async def status():
    return {
        "api": "online",
        "ia": provider.nome if provider else "não configurado",
        "ia_disponivel": ia_disponivel,
        "tipos_suportados": TIPOS_DOCUMENTOS,
        "formatos_aceitos": sorted(EXTENSOES_PERMITIDAS),
        "tamanho_maximo_mb": MAX_FILE_SIZE // (1024 * 1024),
    }


@app.post("/processar", response_model=RespostaProcessamento)
async def processar_documentos(arquivos: list[UploadFile] = File(...)):
    """
    Processa múltiplos documentos com Ollama.
    Aceita: PDF, JPG, PNG, WEBP, BMP, TIFF
    """
    if not ia_disponivel or provider is None:
        raise HTTPException(status_code=503, detail={
            "erro": "Ollama não disponível",
            "mensagem": (
                "Certifique-se de que o Ollama está instalado e em execução, "
                "e que o modelo de visão está instalado (ollama pull llama3.2-vision)."
            )
        })

    if not arquivos:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    if len(arquivos) > 20:
        raise HTTPException(status_code=400, detail="Máximo de 20 arquivos por requisição")

    resultados: list[ResultadoArquivo] = []
    tipos_encontrados: set[str] = set()

    for arquivo in arquivos:
        nome = arquivo.filename or "arquivo_sem_nome"
        logger.info(f"Processando: {nome}")

        extensao = Path(nome).suffix.lower()
        if extensao not in EXTENSOES_PERMITIDAS:
            resultados.append(ResultadoArquivo(
                nome_arquivo=nome,
                erro=f"Formato não suportado: {extensao}. Use: {', '.join(sorted(EXTENSOES_PERMITIDAS))}",
                reconhecido=False
            ))
            continue

        conteudo = await arquivo.read()

        if len(conteudo) > MAX_FILE_SIZE:
            resultados.append(ResultadoArquivo(
                nome_arquivo=nome,
                erro=f"Arquivo muito grande ({len(conteudo) // (1024*1024)}MB). Máximo: 20MB",
                reconhecido=False
            ))
            continue

        if len(conteudo) == 0:
            resultados.append(ResultadoArquivo(
                nome_arquivo=nome, erro="Arquivo vazio", reconhecido=False
            ))
            continue

        try:
            resultado_doc = processar_documento(conteudo, nome, provider)
            tipo = resultado_doc.get("tipo_documento", "DESCONHECIDO")
            dados = resultado_doc.get("dados", {})
            reconhecido = tipo != "DESCONHECIDO"

            if reconhecido:
                tipos_encontrados.add(tipo)

            resultados.append(ResultadoArquivo(
                nome_arquivo=nome, tipo_documento=tipo, dados=dados, reconhecido=reconhecido
            ))

        except ValueError as e:
            logger.error(f"Erro de validação em {nome}: {e}")
            resultados.append(ResultadoArquivo(nome_arquivo=nome, erro=str(e), reconhecido=False))
        except Exception as e:
            logger.error(f"Erro ao processar {nome}: {e}")
            resultados.append(ResultadoArquivo(
                nome_arquivo=nome,
                erro=f"Erro interno: {str(e)}",
                reconhecido=False
            ))

    faltantes = sorted(set(TIPOS_DOCUMENTOS) - tipos_encontrados)

    return RespostaProcessamento(
        arquivos_processados=resultados,
        documentos_reconhecidos=sorted(tipos_encontrados),
        documentos_faltantes=faltantes,
        total_arquivos=len(arquivos),
        total_reconhecidos=sum(1 for r in resultados if r.reconhecido)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
