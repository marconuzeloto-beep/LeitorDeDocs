"""
API principal do sistema LeitorDeDocs.
Suporta múltiplos provedores de IA: Gemini, Groq (Llama) e Ollama (Llama local).
Configure o provedor desejado com AI_PROVIDER no arquivo .env.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_provider import AIProvider, OllamaIndisponivel, criar_provedor
from document_processor import TIPOS_DOCUMENTOS, processar_documento
from models import RespostaProcessamento, ResultadoArquivo

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 20 * 1024 * 1024
EXTENSOES_PERMITIDAS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

provider: Optional[AIProvider] = None
ia_disponivel: bool = False


def _ler_config() -> dict:
    """Lê as configurações de provedor do ambiente."""
    return {
        "provider":        os.getenv("AI_PROVIDER", "ollama").strip(),
        "gemini_api_key":  os.getenv("GEMINI_API_KEY", "").strip(),
        "groq_api_key":    os.getenv("GROQ_API_KEY", "").strip(),
        "groq_model":      os.getenv("GROQ_MODEL", "").strip(),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        "ollama_model":    os.getenv("OLLAMA_MODEL", "").strip(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider, ia_disponivel

    config = _ler_config()
    nome_provedor = config["provider"]

    # Verifica se as credenciais mínimas estão presentes (Ollama não precisa de chave)
    chave_ausente = (
        (nome_provedor == "gemini" and not config["gemini_api_key"]) or
        (nome_provedor == "groq"   and not config["groq_api_key"])
    )

    if chave_ausente:
        chave_necessaria = "GEMINI_API_KEY" if nome_provedor == "gemini" else "GROQ_API_KEY"
        logger.warning(
            f"⚠️  {chave_necessaria} não configurada para o provedor '{nome_provedor}'. "
            f"O servidor iniciou, mas o processamento estará desativado. "
            f"Configure a chave no arquivo .env."
        )
        ia_disponivel = False
    else:
        try:
            provider = criar_provedor(config)
            ia_disponivel = True
            logger.info(f"✅ Provedor de IA configurado: {provider.nome}")
        except OllamaIndisponivel as e:
            logger.error(f"❌ {e}")
            logger.error(
                "   Verifique se o Ollama está instalado e em execução.\n"
                "   → Instale em: https://ollama.com\n"
                "   → Inicie com: ollama serve\n"
                f"  → Baixe o modelo: ollama pull {config.get('ollama_model') or 'llama3.2-vision'}"
            )
            ia_disponivel = False
        except Exception as e:
            logger.error(f"❌ Erro ao configurar provedor de IA '{nome_provedor}': {e}")
            ia_disponivel = False

    yield
    logger.info("Servidor encerrado")


app = FastAPI(
    title="LeitorDeDocs API",
    description="Extração inteligente de documentos — suporta Gemini, Groq/Llama e Ollama/Llama",
    version="2.0.0",
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
        "versao": "2.0.0",
        "provedor": provider.nome if provider else None,
        "ia_disponivel": ia_disponivel,
        "aviso": None if ia_disponivel else (
            "Provedor de IA não configurado. Verifique AI_PROVIDER e as chaves de API no arquivo .env."
        )
    }


@app.get("/status")
async def status():
    config = _ler_config()
    return {
        "api": "online",
        "provedor_configurado": config["provider"],
        "ia": provider.nome if provider else "não configurado",
        "ia_disponivel": ia_disponivel,
        "tipos_suportados": TIPOS_DOCUMENTOS,
        "formatos_aceitos": sorted(EXTENSOES_PERMITIDAS),
        "tamanho_maximo_mb": MAX_FILE_SIZE // (1024 * 1024),
    }


@app.post("/processar", response_model=RespostaProcessamento)
async def processar_documentos(arquivos: list[UploadFile] = File(...)):
    """
    Processa múltiplos documentos usando o provedor de IA configurado.
    Aceita: PDF, JPG, PNG, WEBP, BMP, TIFF
    """
    if not ia_disponivel or provider is None:
        config = _ler_config()
        nome_prov = config["provider"]
        dicas = {
            "gemini": "Adicione GEMINI_API_KEY no .env — obtenha em https://aistudio.google.com/app/apikey",
            "groq":   "Adicione GROQ_API_KEY no .env — obtenha gratuitamente em https://console.groq.com",
            "ollama": "Instale o Ollama (https://ollama.com) e execute: ollama pull llama3.2-vision",
        }
        raise HTTPException(status_code=503, detail={
            "erro": "Provedor de IA não configurado",
            "mensagem": dicas.get(nome_prov, f"Configure o provedor '{nome_prov}' no arquivo .env.")
        })

    if not arquivos:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    if len(arquivos) > 20:
        raise HTTPException(status_code=400, detail="Máximo de 20 arquivos por requisição")

    resultados: list[ResultadoArquivo] = []
    tipos_encontrados: set[str] = set()

    for arquivo in arquivos:
        nome = arquivo.filename or "arquivo_sem_nome"
        logger.info(f"Processando: {nome} via {provider.nome}")

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
            resultados.append(ResultadoArquivo(nome_arquivo=nome, erro="Arquivo vazio", reconhecido=False))
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
                erro=f"Erro interno ao processar o documento: {str(e)}",
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
